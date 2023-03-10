import pathlib
from typing import List, Callable, Any, Union


def callback_dummy(*args, **kwargs):
    """A dummy function that does nothing"""
    pass

class SmartTask:
    """
    A decorator class that can be used to mark functions/code snippets as tasks, allowing the user to define 
    all input/output files and using a callback_funtion can signal if the task should run.
    
    Usage: 
    - Use this decorator class by calling it on the desired function
    - Optional arguments:
        * input_list : list of str containing the names all input files or of None if the input file is some data in memory.
        * output_list : list of str containing the names all output files.
        * force_rerun : boolean indicating if the task should be run even if the inputs/outputs are up to date.
        * check_timestamps : boolean indicating whether or not to check the last modified timestamp of the files.
        * call_back : callable object that signals whether or not the task should run.def func(input_list, output_list):
        
    Example:
    
    @SmartTask(input_list=['input.test'],output_list=['output.test'], force_rerun=False, check_timestamps=True, call_back=callback_print_should_run)
    def addone(input_list, output_list):
        # External Python function that will be turned into a task via decorator SmartTask
        a = pd.read_csv(input_list[0], names=['data'])
        a['data'] += 1
        a.to_csv(output_list[0], index=False, header=False)
        print('one added')
    
    Note: This decorator class can be used to perform same tasks and avoid re-do the tasks which have been completed already.
          Whether the code will be run or not is decided when the decorated code is defined, not when it runs. See the example in __main__
    """


    def __init__(self, input_list: List[Union[str,None]], output_list: List[str], force_rerun: bool=False, check_timestamps: bool=True, call_back: Callable[[bool],Any]=callback_dummy):
        """
        Constructor for SmartTask class.

        Args:
            input_list: a list of str containing the names all input files or of None if the input file is some data in memory.
            output_list: a list of str containing the names all output files.
            force_rerun: a boolean indicating if the task should be run even if the inputs/outputs are up to date.
            check_timestamps: a boolean indicating whether or not to check the last modified timestamp of the files.
            call_back: a callable object that signals whether or not the task should run. 
        """
        self.input_list = input_list
        self.output_list = output_list
        self.force_rerun = force_rerun
        self.check_timestamps = check_timestamps
        self.call_back = call_back
        # self.call_back = callback_dummy if call_back is None else call_back


    def file_exists(self, file: str) -> bool:
        """
        Check if a file exists.

        Args:
          file -- the name of the file to be checked

        Returns:
          Boolean indicating whether or not file exists.
        """
        return file is None or pathlib.Path(file).exists()


    def is_output_update_needed(self, input_file: str, output_file: str) -> bool:
        """
        Checks if input and output files exist and if output file needs updating.

        Args:
            input_file -- name of input file to check
            output_file - name of output file to check

        Returns:
            Boolean indicating whether files exist and if output needs updating.
        """
        if not self.file_exists(input_file):
            raise FileNotFoundError(f'Some input file missing: {input_file}')

        if not self.file_exists(output_file):
            return True

        if input_file is None:
            return False

        if self.check_timestamps:
            return pathlib.Path(input_file).stat().st_mtime >= pathlib.Path(output_file).stat().st_mtime

    def should_run(self):
        """
        Check whether or not the task should run.

        Returns:
            A boolean indicating whether the task should run or not.
        """
        return self.force_rerun or any(
            self.is_output_update_needed(input_file, output_file)
            for input_file, output_file in zip(self.input_list, self.output_list)
        )

        
    def _simple_task(self, func):
        """
        The main function that runs the code in either an idle or skipped state.

        If the task should run, then this function executes the input function and passes all the required inputs.
        Otherwise, it just ignores the input function and does nothing.

        Args:
        func: the input function to be called if the task needs to be run

        Returns:
        func_run or func_skip depending on whether or not the task should run.
        """
        def func_run(*args, **kwargs):
            kwargs['input_list'] = self.input_list
            kwargs['output_list'] = self.output_list
            func(*args, **kwargs)

        def func_skip(*args, **kwargs):
            pass

        should_run = self.should_run()
        self.call_back(should_run)
        return func_run if should_run else func_skip

    def __call__(self, func):
        """
        Override the instance being called as a function on an external function.
         
        Args:
            func: An external Python function to be turned into a task using this decorator class. 
        
        Returns:
            If the task should execute, returns the original function with optional additional arguments; otherwise
            returns a do-nothing function.
        """
        return self._simple_task(func)





##  test  ##
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    a = pd.DataFrame(np.arange(10).reshape(-1,1))
    a.to_csv('input.test', index=False, header=False)


    def callback_print_should_run(x):
        print('should_run',x)

    @SmartTask(input_list=['input.test'],output_list=['output.test'], force_rerun=False, check_timestamps=True, call_back=callback_print_should_run)
    def addone(input_list, output_list):
        a = pd.read_csv(input_list[0], names=['data'])
        a['data']+=1
        a.to_csv(output_list[0], index=False, header=False)
        print('one added')


    addone() ## this will run
    addone() ## this will run as well

    @SmartTask(input_list=['input.test'],output_list=['output.test'], force_rerun=False, check_timestamps=True, call_back=callback_print_should_run)
    def addoneagain(input_list, output_list):
        a = pd.read_csv(input_list[0], names=['data'])
        a['data']+=1
        a.to_csv(output_list[0], index=False, header=False)
        print('one added again')

    addoneagain() ## this will not run

