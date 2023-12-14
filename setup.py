from distutils.core import setup, Extension
import sysconfig

def main():
    CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
    LDFLAGS = ['-L/usr/lib/llvm-10/lib','-fopenmp']
    
    module1 = Extension('numc',
                    include_dirs=['matrix.h'],
                    extra_link_args=LDFLAGS,
                    sources=['numc.c','matrix.c'],
                    extra_compile_args=CFLAGS)
    '''
    我们要做的不是include_dirs = ['/usr/local/include'],
                libraries = ['tcl83'],
                library_dirs = ['/usr/local/lib'],
    这些文档里的用例, 居然只需要extra_compile_args,extra_link_args，感觉很神奇
    '''
    setup(name="numc",
          version='1.0',
          description='This is a demo package',
          ext_modules=[module1])  

if __name__ == "__main__":
    main()
