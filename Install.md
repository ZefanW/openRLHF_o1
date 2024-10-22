## Install
安装Deepspeed
```bash
DS_BUILD_CPU_ADAM=1 python -m pip install deepspeed==0.14.2
# AttributeError: ‘DeepSpeedCPUAdam‘ object has no attribute ‘ds_opt_adam
# 参考blog：https://blog.csdn.net/qq_44193969/article/details/137051032
python -c 'import deepspeed; deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()'
```

## Run
