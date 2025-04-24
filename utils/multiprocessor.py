import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def local_mt(datums: list, func: callable, desc: str = None, num_workers=16):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        if desc is None:
            return list(executor.map(func, datums))
        return list(tqdm.tqdm(executor.map(func, datums), total=len(datums), desc=desc))
    
def local_mp(datums: list, func: callable, desc: str = None, num_workers=16):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        if desc is None:
            return list(executor.map(func, datums))
        return list(tqdm.tqdm(executor.map(func, datums), total=len(datums), desc=desc))


