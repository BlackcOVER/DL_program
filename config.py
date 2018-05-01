import argparse 

parse = argparse.ArgumentParser()

parse.add_argument('--input_dir', type=str, default='/home/albert/MNIST_data/')
parse.add_argument('--batch_size', type=int, default=128)
parse.add_argument('--d_learning_rate', type=float, default=0.0002)
parse.add_argument('--g_learning_rate', type=float, default=0.0002)
parse.add_argument('--max_epochs', type=int, default=2)
parse.add_argument('--fake_data', default=False)
parse.add_argument('--example_num', type=int, default=200)
parse.add_argument('--ckpt_dir', type=str, default='/home/albert/gan/save/')
parse.add_argument('--dc_ckpt_dir', type=str, default='/home/albert/gan/dc_save/')
parse.add_argument('--smooth', type=float, default=0.1)
FLAGS, unparsed = parse.parse_known_args()

if __name__ == '__main__':
    print FLAGS



