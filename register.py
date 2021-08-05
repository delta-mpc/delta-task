from delta import DeltaNode
from delta.task import LearningTask
from task import preprocess, train_step

if __name__ == "__main__":
    delta_node = DeltaNode("http://127.0.0.1:6700")
    task = LearningTask(
        name="test",
        dataset="mnist.npz",
        preprocess=preprocess,
        train_step=train_step,
        dataloader={"batch_size": 10, "shuffle": True, "drop_last": True},
        total_epoch=10,
        members=["2", "3"]
    )
    delta_node.create_task(task)
