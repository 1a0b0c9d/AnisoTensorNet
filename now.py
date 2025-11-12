# import optuna

# # 加载已有的 Optuna study
# study = optuna.load_study(
#     study_name="matbench_v0.1_AnisoTensorNet_9_22",
#     storage="sqlite:///aniso_optuna.db",
# )

# # 输出当前最佳 trial 的相关信息
# best_trial = study.best_trial
# print(f"Best trial number: {best_trial.number}")
# print(f"Best trial value (val_loss): {best_trial.value}")
# print(f"Best trial parameters: {best_trial.params}")

# # 如果想查看所有的试验结果
# print("\nAll trials:")
# for trial in study.trials:
#     print(f"Trial #{trial.number}: state={trial.state} value={trial.value} params={trial.params}")
import optuna

study = optuna.load_study(
    study_name="matbench_v0.1_AnisoTensorNet_9_22",
    storage="sqlite:///aniso_optuna.db",
)

# 停止当前优化进程
study.stop()  # 暂停当前的优化
print("Optimization paused.")
