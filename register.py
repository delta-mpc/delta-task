from delta import Registry
from delta.task import LearningTask
from task import preprocess, train_step

if __name__ == "__main__":
    registry = Registry("http://127.0.0.1:6700")
    task = LearningTask(
        name="test",
        dataset="mnist.npz",
        preprocess=preprocess,
        train_step=train_step,
        dataloader={"batch_size": 10, "shuffle": True, "drop_last": True},
        total_epoch=100,
        members=["2", "3"]
    )
    registry.create_task(task)
