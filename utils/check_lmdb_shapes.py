import lmdb
import argparse
import numpy as np
def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        shape_str = txn.get(f"{array_name}_shape".encode()).decode()
        shape = tuple(map(int, shape_str.split()))
    return shape

def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, row_index, shape=None):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    if dtype == str:
        array = row_bytes.decode()
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)

    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to the LMDB file")
    args = parser.parse_args()

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    latents_shape = get_array_shape_from_lmdb(env, 'latents')
    # latent = retrieve_row_from_lmdb(env, 'latents', np.float16, 0)
    
    print(f"latents shape: {latents_shape}")

if __name__ == "__main__":
    main() 