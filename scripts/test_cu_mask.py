#!/usr/bin/env python
# This script tests performance under various compute unit mask settings.
import argparse
import json
import subprocess
import math
import pysmctrl
"""
if __name__ == "__main__":
    for dev in range(2):
        info = pysmctrl.get_gpc_info(dev)
        print("GPU %d has %d GPCs"%(dev, len(info)))
        for i in range(len(info)):
            print("Mask of TPCS associated with GPC %d: %#016lx"%(i, info[i]))
    exit()
"""
def generate_config(cu_mask, striped):
    """ Returns a JSON string containing a config. The config will use the
    Matrix Multiply plugin with a 1024x1024 matrix, using 32x32 thread blocks.
    Only the compute unit mask is varied.
    """
    active_cu_count = bin(~cu_mask).count("1") # Count num _enabled_ TPCs
    hex_mask = cu_mask_to_hex_string(cu_mask)
    plugin_config = {
        "label": str(active_cu_count),
        "log_name": "cu_mask_%s.json" % ("striped_" + hex_mask if striped else hex_mask),
        "filename": "./bin/matrix_multiply.so",
        "thread_count": [32, 32],
        "block_count": 1,
        "sm_mask": hex_mask,
        "data_size": 0,
        "additional_info": {
            "matrix_width": 1024,
            "skip_copy": True
        }
    }
    name = "Compute Unit Count vs. Performance"
    # Indicate the stripe width if it isn't the default
    if striped:
        name += " (striped)"
    overall_config = {
        "name": name,
        "max_iterations": 100,
        "max_time": 0,
        "cuda_device": 0,
        "pin_cpus": True,
        "do_warmup": True,
        "benchmarks": [plugin_config]
    }
    return json.dumps(overall_config)

def cu_mask_to_binary_string(int_cu_mask):
    """ A utility function that takes an array of booleans and returns a string
    of 0s and 1s. """
    return format(int_cu_mask, "064b")

def cu_mask_to_hex_string(int_cu_mask):
    return "%016x" % (int_cu_mask & 0xffffffffffffffff)

def run_process(int_cu_mask, striped):
    """ This function starts a process that will run the plugin with the given
    compute unit mask. Also requires a stripe width to include to use in the
    labeling of output files. """
    config = generate_config(cu_mask, striped)
    print("Starting test with CU mask " + cu_mask_to_hex_string(int_cu_mask))
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

def get_cu_mask(active_cus, total_cus, striped, dev):
    """ Returns a CU mask (represented as a bitstring) with the active number
    of CUs specified by active_cus, striping across GPCs if requested. """
    res = long(0)
    if not striped:
        for b in range(active_cus):
            res = res << 1
            res |= 1
    else:
        info = pysmctrl.get_gpc_info(dev)
        next_gpc = 0
        for b in range(active_cus):
            # Get rightmost set bit (next available TPC for this GPC)
            # If no TPC is available in this GPC, look in the next one.
            next_bit = 0
            while next_bit == 0:
                mask = info[next_gpc]
                next_bit = mask & (-mask)
                next_gpc = (next_gpc + 1) % len(info)
            # Add to mask of enabled TPCs
            res |= next_bit
            # Remove that TPC from the available list
            info[next_gpc-1] -= next_bit
#        print("GPU %d has %d GPCs" % (0, len(info)))
#        for i in range(len(info)):
#            print("Mask of TPCS associated with GPC %d: %#016lx" % (i, info[i]))
    return ~res # As SM mask is opposite, invert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cu_count", type=int,
        help="The total number of CUs on the GPU.")
    parser.add_argument("--start_count", type=int, default=0,
        help="The number of CUs to start testing from. Can be used to resume "+
            "tests if one hung.")
    parser.add_argument("-s", "--stripe", action="store_true",
        help="If pysmctrl is available, \"stripe\" TPC assignment using the "+
            "CU-distributed algorithm.")
    parser.add_argument("-d", "--device", type=int, default=0,
        help="Requires \"-s\". Which GPU to pull striping configuration for")
    args = parser.parse_args()
    if args.cu_count:
        cu_count = args.cu_count
        if (cu_count <= 0):
            print("The CU count must be positive.")
    elif pysmctrl:
        cu_count = pysmctrl.get_tpc_info(args.device)
        print("Auto-detected a cu_count of %d" % cu_count)
    if (pysmctrl and cu_count > pysmctrl.get_tpc_info(args.device)):
        print("The CU count must not exceed the number of available TPCs.")
    if (args.stripe and not pysmctrl):
        print("To use accurate striping, pysmctrl must be available and "+
              "nvdebug must be loaded")
        exit(1)
    for active_cus in range(args.start_count, cu_count):
        print("Running test for %d (+ 1) active CUs." % (active_cus))
        cu_mask = get_cu_mask(active_cus + 1, cu_count, args.stripe, args.device)
        run_process(cu_mask, args.stripe)
        print("\n")

