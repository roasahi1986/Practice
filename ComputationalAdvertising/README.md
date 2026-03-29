# model_server

## preparation

Prior to commencing the project build, kindly ensure that you follow the steps outlined in scirpt/prepare.sh to set up the compilation environment accordingly.

## check

* Enable bazel clean before build (which is disabled by default):
```shell
$ CLEAN=true ./build.sh
```

* Disable static code check (which is enabled by default):
```shell
$ STATIC_CODE_CHECK=false ./build.sh
```

* Enable unit-test and benchmark-test (which is disabled by default):
```shell
$ UNIT_TEST=true BENCHMARK_TEST=true ./build.sh
```

