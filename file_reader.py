from glob import glob

def read_files():
    files = glob('./stride_data/data/*')
    print(files)

if __name__ == '__main__':
    read_files()