import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def run_mp(
    map_func,
    arg_list,
    combine_func=None,
    num_cores=-1,
    split_arg=-1,
    progress_bar=True,
    **kwargs
):
    """multiprocessing function to run map_func in parallel

    Args:
        map_func (func): function to be run in parallel
        arg_list (List): list of arguments to be passed to map_func
        combine_func (func, optional): function to combine the results from map_func. Defaults to None.
        num_cores (int, optional): The number of cores to be used. Defaults to -1.
        split_arg (int, optional): The number of chunks to split the arg_list. Defaults to -1.
        progress_bar (bool, optional): progress bar. Defaults to False.
        **kwargs: other constant arguments to be passed to map_func

    Returns:
        List: results from map_func and combine_func
    """
    if num_cores == -1:
        num_cores = mp.cpu_count()
        num_cores = len(arg_list) if len(arg_list) < num_cores else num_cores

    if split_arg != -1:
        arg_list = list(chunks(arg_list, split_arg))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as pool:
        # For multithreading (actual code might be different)
        # See here: https://docs.python.org/3/library/threading.html
        #     with concurrent.futures.ThreadPoolExecutor(max_threads=num_threads) as pool:
        with tqdm(total=len(arg_list)) as progress:
            futures = []

            for args in arg_list:
                # the arg list need to be created if there are more than one arguments
                future = pool.submit(map_func, args, **kwargs)
                if progress_bar:
                    future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            results = []

            for future in futures:
                result = future.result()
                results.append(result)

    if combine_func is not None:
        return combine_func(results)
    else:
        return results


def add_num(a, b=1):
    return a + b


def add_num_2(a):
    return len(a)


if __name__ == "__main__":
    args_list = list(range(100))

    # run normal map function with a list of arguments
    print(run_mp(add_num, args_list, progress_bar=True, num_cores=10))

    # run normal map function with a list of arguments and combine the results
    print(run_mp(add_num, args_list, combine_func=sum, progress_bar=True, num_cores=10))

    # run normal map function with a list of arguments and constant arguments
    print(
        run_mp(
            add_num, args_list, combine_func=sum, progress_bar=True, num_cores=10, b=10
        )
    )

    # split the arguments list into chunks, this function should be able to handle list as arguments
    print(run_mp(add_num_2, args_list, split_arg=10, progress_bar=True, num_cores=10))
