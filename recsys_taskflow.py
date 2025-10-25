"""
recsys_taskflow.py

一个真实可运行的“推荐系统训练任务调度与执行”示例。

功能：
- 每天凌晨 2:00 自动发布模型训练任务；
- Producer 将任务发送到 Kafka；
- Consumer 异步消费任务；
- Executor 执行推荐模型训练；
- Redis 存储任务状态；
- Scheduler 控制任务调度时间。

依赖：
pip install kafka-python redis apscheduler scikit-learn pandas numpy
"""

import threading
import time
import json
import random
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

from apscheduler.schedulers.background import BackgroundScheduler
from kafka import KafkaProducer, KafkaConsumer
import redis


# ====== 全局配置 ======
KAFKA_TOPIC = "recsys_task"
KAFKA_BOOTSTRAP = "localhost:9092"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="recsys_consumers",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)


# ====== 模型定义 ======
class RecsysModel:
    """
    一个简单的矩阵分解推荐模型 (TruncatedSVD 实现)。

    用于模拟推荐系统训练与推荐阶段的过程。

    Attributes:
        n_components (int): 潜在因子维度。
        model (TruncatedSVD): sklearn 模型实例。
        user_index (Index): 用户索引。
        item_index (Index): 物品索引。
        latent (ndarray): 用户潜在向量矩阵。
    """

    def __init__(self, n_components: int = 10):
        """
        初始化模型。

        Args:
            n_components (int): 潜在因子维度。
        """
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components)

    def train(self, df: pd.DataFrame):
        """
        训练矩阵分解模型。

        Args:
            df (pd.DataFrame): 包含列 ["user", "item", "rating"] 的评分数据。

        Returns:
            RecsysModel: 返回自身对象。

        Example:
            >>> df = pd.DataFrame({"user":[0,0,1], "item":[1,2,2], "rating":[5,3,4]})
            >>> model = RecsysModel().train(df)
        """
        pivot = df.pivot(index="user", columns="item", values="rating").fillna(0)
        self.user_index = pivot.index
        self.item_index = pivot.columns
        self.latent = self.model.fit_transform(pivot)
        return self

    def recommend(self, user_id: int, top_k: int = 5):
        """
        根据训练好的模型，为指定用户推荐物品。

        Args:
            user_id (int): 用户 ID。
            top_k (int): 推荐物品数量。

        Returns:
            list[int]: 推荐物品 ID 列表。

        Example:
            >>> model.recommend(12, top_k=3)
            [102, 59, 218]
        """
        if user_id not in self.user_index:
            return []
        user_vec = self.latent[self.user_index.get_loc(user_id)]
        item_latent = self.model.components_.T
        scores = np.dot(item_latent, user_vec)
        top_items = self.item_index[np.argsort(scores)[::-1][:top_k]]
        return list(map(int, top_items))


def generate_fake_data(n_users: int = 2000, n_items: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟的 user-item-rating 数据集。

    Args:
        n_users (int): 用户数。
        n_items (int): 物品数。
        seed (int): 随机种子。

    Returns:
        pd.DataFrame: 包含 user, item, rating 三列的数据。

    Example:
        >>> df = generate_fake_data(3, 5)
        >>> df.head(2)
           user  item  rating
        0     1     4       3
        1     2     0       5
    """
    np.random.seed(seed)
    users = np.random.randint(0, n_users, size=100000)
    items = np.random.randint(0, n_items, size=100000)
    ratings = np.random.randint(1, 6, size=100000)
    return pd.DataFrame({"user": users, "item": items, "rating": ratings})


# ====== Producer ======
def publish_task(task_type: str = "train"):
    """
    向 Kafka 发布新的训练任务。

    Args:
        task_type (str): 任务类型，例如 "train"、"evaluate"。

    Example:
        >>> publish_task("train")
        [Producer] 🟢 Published train task: train_1730512800
    """
    task_id = f"{task_type}_{int(time.time())}"
    task = {
        "id": task_id,
        "type": task_type,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 避免重复任务
    if not rds.get(f"lock:{task_id}"):
        rds.set(f"lock:{task_id}", "1", ex=300)
        producer.send(KAFKA_TOPIC, value=task)
        producer.flush()
        rds.hset("tasks", task_id, "published")
        print(f"[Producer] 🟢 Published {task_type} task: {task_id}")
    else:
        print(f"[Producer] ⏸ Skipping duplicate task: {task_id}")


# ====== Executor ======
def execute_task(task: dict):
    """
    执行训练任务（模拟训练推荐模型）。

    Args:
        task (dict): 任务字典，包含 id, type, created_at。

    Example:
        >>> execute_task({"id":"train_1730512800","type":"train"})
        [Executor] 🚀 Running task train_1730512800
        [Executor] ✅ Completed train_1730512800 (took 3.25s)
    """
    task_id = task["id"]
    print(f"[Executor] 🚀 Running task {task_id}")
    start_time = time.time()

    try:
        df = generate_fake_data()
        model = RecsysModel().train(df)
        user_id = random.choice(df["user"].unique())
        rec_items = model.recommend(user_id)

        elapsed = round(time.time() - start_time, 2)
        result = {
            "status": "success",
            "duration": elapsed,
            "user": int(user_id),
            "rec_items": rec_items[:5],
        }

        rds.hset("tasks", task_id, json.dumps(result))
        print(f"[Executor] ✅ Completed {task_id} (took {elapsed}s)")

    except Exception as e:
        rds.hset("tasks", task_id, json.dumps({"status": "error", "msg": str(e)}))
        print(f"[Executor] ❌ Error in {task_id}: {e}")


# ====== Consumer ======
def consume_tasks():
    """
    持续消费 Kafka 中的任务并调用 Executor 执行。

    Example:
        >>> consume_tasks()
        [Consumer] 🟦 Waiting for tasks ...
    """
    print("[Consumer] 🟦 Waiting for tasks ...")
    for msg in consumer:
        task = msg.value
        execute_task(task)


# ====== Scheduler ======
def start_scheduler():
    """
    启动 APScheduler，每天凌晨 2:00 触发训练任务发布。

    Example:
        >>> start_scheduler()
        [Scheduler] ⏰ Scheduler started (daily at 2:00 AM)
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(publish_task, "cron", hour=2, minute=0, kwargs={"task_type": "train"})
    scheduler.start()
    print("[Scheduler] ⏰ Scheduler started (daily at 2:00 AM)")


# ====== 主程序入口 ======
if __name__ == "__main__":
    print("🔥 Starting RecSys TaskFlow Engine")

    # 启动 Scheduler
    threading.Thread(target=start_scheduler, daemon=True).start()

    # 启动 Consumer
    threading.Thread(target=consume_tasks, daemon=True).start()

    # 主线程监控任务状态
    try:
        while True:
            time.sleep(30)
            print("\n[Monitor] Redis task snapshot:")
            tasks = rds.hgetall("tasks")
            for k, v in list(tasks.items())[-5:]:
                print(f"  {k}: {v}")
            print("-" * 60)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ...")
