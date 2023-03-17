import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.sparse import issparse
import anndata
from statsmodels.stats.multitest import fdrcorrection
from numba import jit
from ..utils.group_obs_mean import group_obs_mean
from collections.abc import Iterable
from typing import Union

@jit
def fast_auc(y_true:Iterable[bool], y_prob:Iterable[float]) -> np.array:
    """
    Compute the area under the curve (AUC) score for multiple features.

    Args:
        y_true: 1d binary labels of size m.
        y_prob: Predicted probability matrix of size m x n.

    Returns:
        AUC scores for each predicted probability set. In total, n AUC scores will be calculated.
    """
    if issparse(y_prob):
        y_prob = y_prob.todense()
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_true = y_true[np.argsort(y_prob,1)]
    nobs,nvar = y_prob.shape
    
    aucs = np.zeros(nobs)
    nfalse = np.zeros(nobs)
    for j in range(nobs):
        for i in range(nvar):
            nfalse[j] += (1 - y_true[j, i])
            aucs[j] += y_true[j, i] * nfalse[j]
#         aucs[j] /= (nfalse[j] * (nvar - nfalse[j]))
    aucs /= nfalse * (nvar - nfalse)
    return aucs

def welch_t(mean1:float, mean2:float, sesq1:float, sesq2:float, n1:int, n2:int) -> float:
    """
    Perform Welch's t-test to compare means from two independent group_names with unequal variances.

    Args:
        mean1: Mean of group 1.
        mean2: Mean of group 2.
        sesq1: Squared standard error of group 1.
        sesq2: Squared standard error of group 2.
        n1: Number of samples in group 1.
        n2: Number of samples in group 2.

    Returns:
        The p-value for the two-sided test.
    """
    t = (mean1-mean2)/((sesq1+sesq2)**0.5)
    v = (sesq1+sesq2)**2/( sesq1**2/(n1-1) + sesq2**2/(n2-1) )
    
    ## use sf instead of cdf, which is not accurate enough
    # https://stackoverflow.com/questions/6298105/precision-of-cdf-in-scipy-stats
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    
    #still has problem
    
#     p = (1 - ss.t.cdf(abs(t), v)) * 2
    p = ss.t.sf(abs(t), v) * 2
    return p
    
def nafdrcorrection(p:Iterable[float], alpha:float=0.05, method:str='indep', is_sorted:bool=False) -> tuple:
    """
    Apply the false discovery rate correction on p-values.

    Args:
        p: p-values to be corrected.
        alpha: Significance level.
        method: Method of correction. Default is 'indep'.
        is_sorted: Whether input array is sorted.

    Returns:
        Tuple containing a bool array for whether p-values are significant and another array for their adjusted values.
    """
    ids = ~np.isnan(p)
    adjp = np.array([np.nan]*len(p))
    sig = np.array([False]*len(p))
    fdr_rlt = fdrcorrection(p[ids],alpha,method,is_sorted)
    sig[ids] = fdr_rlt[0]
    adjp[ids] = fdr_rlt[1]
    return sig, adjp
      
class FastMarkerCaller:
    """
    A class FastMarkerCaller that performs independent two-sample t-test and returns significant markers 
    among user-defined groups
    
    Args:
        adata: AnnData object, contains gene expression data, with genes in row and cells in columns.
        groupby: str, column key name for a categorical grouping information of cells.

    Attributes:
        sizes: The number of cells in each group. Size of each group provided by groupby variable.
        means: Mean expression level of each gene across cell of the same group list.
        stds: Standard deviations of gene expression values within a given group.
        sqstderrs: Mean squared standard error for each gene.
        feat_names: Names of the features (e.g., genes)
        group_names: Names of the groups obtained from `groupby`
        adata: AnnData object used in instantiating the object.
        groupby: name of the grouping variable used in instantiating the object.

    Methods:
        call_markers: Perform Welch's independent two-sample t-test on different groups and returns 
            significant upregulated/deactivated gene(s)

    Example:
    #### Initialize FastMarkerCaller object on example dataset ####
    fastcaller = FastMarkerCaller(adata, 'batch')
    #### Using the object to perform marker test ####
    res = fastcaller.call_markers(['batch1'], ['batch2'], fdr=0.05)
    string_res = res.to_string()
    print(string_res)
        
    """
    def __init__(self, adata:anndata.AnnData, groupby:str, ) -> None:
        self.sizes = adata.obs[groupby].value_counts(sort=False)
        self.means = group_obs_mean(adata, groupby)[self.sizes.index]
        self.stds = group_obs_mean(adata, groupby, np.std)[self.sizes.index]
        self.sqstderrs = self.stds**2/self.sizes

        self.feat_names = adata.var_names
        self.group_names = self.sizes.index
        
        self.adata = adata # save for auroc 
        self.groupby = groupby
        
    def _grp_info(self, groups:Iterable[str]) -> tuple:
        """Extract group sizes, group means, and group squared standard errors"""

        ind_sizes = self.sizes.loc[groups]
        ind_means = self.means[groups]
        ind_sqstderrs = self.sqstderrs[groups]
        mg_mean = (ind_means*ind_sizes).sum(1)/ind_sizes.sum()    
        mg_sqstderr = (ind_sqstderrs*ind_sizes*(ind_sizes-1)).sum(1)/sum(ind_sizes)/sum(ind_sizes)
        mg_size = ind_sizes.sum()
        return mg_size, mg_mean, mg_sqstderr
        
    
    def call_markers(self, groups1:Union[Iterable,str], groups2:Union[Iterable,str]=None, *, 
                     fdr:float=0.05, topn:int=None, two_sided:bool=False, 
                     auroc:bool=True, auroc_cutoff:float=0.60,) -> pd.DataFrame:
        
        """
        Parameters:
            groups1 (iterable or str): One or more groups for the first category.
            groups2 (iterable or str or None): One or more groups for second group. If not provided
                all other groups will be considered for comparison.
            fdr (float): False discovery rate cut-off. Default is 0.05.
            topn (int): Number of top significant genes returned. Default is None (all significant).
            two_sided (boolean): Whether to look at two-sided t-test or one-sided (over-expressed).
                Default is False. 
            auroc (boolean): Whether to calculate AUROC scores for significant genes using 
                the entire dataset. Default is True.
            auroc_cutoff (float): Cutoff for the minimum value of AUROC score. Default is 0.60.

        Returns:
            pandas.DataFrame: A DataFrame containing summary of markers determined.
            Rows may contain the following columns: p_val, adj_pval, diff, fc, significant and auroc

        """
        if isinstance(groups1, str) or not isinstance(groups1, Iterable):
            groups1 = [groups1]
        if groups2 is None:
            groups2 = self.group_names[~self.group_names.isin(groups1)].tolist()
        if isinstance(groups2, str):
            groups2 = [groups2]
        grps1_size, grps1_mean,grps2_sqstderr = self._grp_info(groups1)
        grps2_size, grps2_mean,grps2_sqstderr = self._grp_info(groups2)
        
        p = welch_t(grps1_mean, grps2_mean, grps2_sqstderr, grps2_sqstderr, 
                    grps1_size, grps2_size)
        sig, adjp = nafdrcorrection(p, fdr)
        
        delta = grps1_mean-grps2_mean
        fc = grps1_mean/grps2_mean
        
        summary = pd.DataFrame([p,adjp,delta,fc,sig], 
                               index=['p-val','adj_p-val','diff','fc','significant'], 
                               columns=self.feat_names).T
        summary = summary.sort_values('adj_p-val')
        summary = summary[summary['adj_p-val']<=fdr]
        if not two_sided:
            summary = summary[summary['diff']>0]
        if topn is not None:
            summary = summary.head(topn)
            
        if auroc:
            adata = self.adata[self.adata.obs[self.groupby].isin(groups1+groups2)]
            isgroups1 = adata.obs[self.groupby].isin(groups1)
            summary['AUROC'] = fast_auc(isgroups1, adata[:, summary.index].X.T)
            summary = summary[summary['AUROC']>auroc_cutoff]
            
        return summary
        
