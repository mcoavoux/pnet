
from joblib import Parallel, delayed

import os
import subprocess


def generate_command_lines(args):
    
    options_tpl = "-i {i} -I {I} -L {L} -l {l} -w {w} -W {W} {D}"
    
    
    for l in args.dim_hidden:
      for L in args.hidden_layers:
        for w in args.dim_word:
          for W in args.dim_wrnn:
            
            others = ""
            if args.use_demographics:
                others += " -D "
            
            if not args.data.startswith("tp"):
                others += " -k {} ".format(args.num_NE)
            
            
            options_all = options_tpl.format(i=args.iterations, I=args.iterations_adv, L=L, l=l, w=w, W=W, D=others)
            
            ptraining = ["--ptraining --alpha {}".format(a) for a in args.alpha]
            
            for o_opts in ["", "--atraining", "--generator char"] + ptraining:
                
                options = options_all + "  " + o_opts
                
                output = "{}/{}".format(args.output, options.replace(" ", "_").replace("-", "_"))
                
                command_line = "python3 main.py {output} {dataset} {options} > {output}_log"
                command_line = command_line.format(output=output, dataset=args.data, options=options)
                
                yield command_line


def unix(command) :
    print(command)
    subprocess.call([command], shell=True)


def main(args):
    os.makedirs(args.output, exist_ok=True)
    Parallel(n_jobs=args.threads)(delayed(unix)(p) for p in generate_command_lines(args))


if __name__ == "__main__":
    
    import argparse
    
    usage="""Launch parallel multiple experiments."""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("data", type=str, choices=["ag", "dw", "tp_fr", "tp_de", "tp_dk", "tp_us", "tp_uk"], help="dataset")
    parser.add_argument("output", type=str, help="output folder")
    
    parser.add_argument("--iterations", "-i", type=int, default=10, help="Number of iterations per experiment")
    parser.add_argument("--iterations-adv", "-I", type=int, default=20, help="Number of iterations for adversary")
    parser.add_argument("--threads", "-N", type=int, default=1, help="Max number of experiments in parallel")
    
    parser.add_argument("--hidden-layers", "-L", type=int, nargs="+", default=[2], help="Number of hidden layers")
    parser.add_argument("--dim-hidden", "-l", type=int, nargs="+", default=[128], help="Size of hidden layers")
    
    parser.add_argument("--dim-word","-w", type=int, nargs="+", default=[32], help="Dimension of word embeddings")
    parser.add_argument("--dim-wrnn","-W", type=int, nargs="+", default=[32], help="Dimension of word lstm")
    
    parser.add_argument("--use-demographics", "-D", action="store_true", help="use demographic variables as input to bi-lstm")
    
    parser.add_argument("--num-NE", "-k", type=int, default=4, help="Number of named entities")
    
    parser.add_argument("--atraining", action="store_true", help="Anti-adversarial training with conditional distribution blurring training")
    parser.add_argument("--ptraining", action="store_true", help="Anti-adversarial training with conditional distribution blurring training")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.1], help="scaling value for anti adversary loss")
    
    args = parser.parse_args()
    
    main(args)














