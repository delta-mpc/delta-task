from task import preprocess, train_step
from delta.task import LearningTask


if __name__ == "__main__":
    task = LearningTask(
        name="test",
        dataset="mnist.npz",
        preprocess=preprocess,
        train_step=train_step,
        dataloader={"batch_size": 10, "shuffle": True, "drop_last": True},
        total_epoch=2,
    )
    with open("task.cfg", mode="wb") as f:
        task.dump_cfg(f)
