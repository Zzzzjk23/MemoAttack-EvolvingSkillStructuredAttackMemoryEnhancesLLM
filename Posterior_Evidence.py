from typing import Optional, Any, List, Set, Tuple, Dict
import pickle
import os
import math
import random

import numpy as np

from prompt_embedding import embed_prompt

attack_methods_list = [
    'Payload Splitting', 'Base64/Hex Encoding', 'Linux Terminal Simulation', 'Character Roleplay', 
    'Language Shifting', 'Cognitive Pressure', 'Scientific/Academic Framing', 'Virtual Machine Escape',
    'Refusal Suppression', 'Game Theory Simulation'
]

class AttackMethod:
    """
    攻击方法类
    
    用于记录和管理具体的攻击方法。
    记录成功次数、失败次数、prompt embedding 向量以及例子。
    """
    
    @staticmethod
    def _array_to_tuple(arr: np.ndarray) -> Tuple[float, ...]:
        """将 numpy 数组转换为可哈希的元组"""
        return tuple(arr.tolist())
    
    @staticmethod
    def _tuple_to_array(t: Tuple[float, ...]) -> np.ndarray:
        """将元组转换回 numpy 数组"""
        return np.array(t)
    
    def __init__(self):
        self.success_count: int = 0
        self.failure_count: int = 0
        self.prompt_embeddings: Set[Tuple[float, ...]] = set()
        self.examples: List[Tuple[np.ndarray, str, str]] =[]

    def update_success(
        self,
        prompt: str,
        example1: str,
        example2: str,
    ) -> None:
        self.success_count += 1
        embedding = embed_prompt(prompt)
        if embedding is not None:
            self.prompt_embeddings.add(self._array_to_tuple(embedding))
            self.examples.append((embedding, example1, example2))
    
    def update_failure(self) -> None:
        self.failure_count += 1
    
    def get_success_probability(self) -> float:
        return self.success_count / (self.success_count + self.failure_count)
    
class PromptCategory:
    def __init__(self):
        self.attack_methods_dict: Dict[str, AttackMethod] = {}
        for method_name in attack_methods_list:
            self.attack_methods_dict[method_name] = AttackMethod()
        self.total_count: int = 0
    def update_success(
        self,
        attackmethod_name: str,
        prompt: str,
        example1: str,
        example2: str,
    ) -> None:
        self.attack_methods_dict[attackmethod_name].update_success(prompt, example1, example2)
        self.total_count += 1
    def update_failure(self, attackmethod_name: str) -> None:
        self.attack_methods_dict[attackmethod_name].update_failure()
        self.total_count += 1
    def calculate_max_cosine_similarity(self, prompt: str, attackmethod_name: str) -> float:
        attack_method = self.attack_methods_dict[attackmethod_name]
        input_embedding = embed_prompt(prompt)
        if input_embedding is None:
            return -1.0
        max_cosine = -1.0
        for existing_embedding_tuple in attack_method.prompt_embeddings:
            existing_embedding = attack_method._tuple_to_array(existing_embedding_tuple)
            dot_product = np.dot(input_embedding, existing_embedding)
            norm_input = np.linalg.norm(input_embedding)
            norm_existing = np.linalg.norm(existing_embedding)
            if norm_input > 0 and norm_existing > 0:
                cosine_sim = dot_product / (norm_input * norm_existing)
                max_cosine = max(max_cosine, cosine_sim)
        return max_cosine

def _get_pe_save_path() -> str:
    """返回 PosteriorEvidence 持久化文件在工作区根目录的路径"""
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(workspace_root, "posterior_evidence.pkl")


def save_posterior_evidence(pe: "PosteriorEvidence", path: Optional[str] = None) -> str:
    """
    将 PosteriorEvidence 实例保存到本地文件（默认在工作区根目录）。
    使用 pickle 序列化，可通过 load_posterior_evidence 读取。

    Args:
        pe: 要保存的 PosteriorEvidence 实例
        path: 保存路径，为 None 时使用工作区根目录下的 posterior_evidence.pkl

    Returns:
        实际保存的绝对路径
    """
    save_path = path if path is not None else _get_pe_save_path()
    with open(save_path, "wb") as f:
        pickle.dump(pe, f)
    return os.path.abspath(save_path)


def load_posterior_evidence(path: Optional[str] = None) -> Optional["PosteriorEvidence"]:
    """
    从本地文件读取 PosteriorEvidence 实例。
    若文件不存在或反序列化失败，返回 None。

    Args:
        path: 文件路径，为 None 时使用工作区根目录下的 posterior_evidence.pkl

    Returns:
        成功则返回 PosteriorEvidence 实例，否则返回 None
    """
    load_path = path if path is not None else _get_pe_save_path()
    if not os.path.isfile(load_path):
        return None
    try:
        with open(load_path, "rb") as f:
            return pickle.load(f)
    except (pickle.PickleError, OSError):
        return None


class PosteriorEvidence:
    def __init__(self, load_path: Optional[str] = None):
        """
        若 load_path 或默认持久化文件存在且可读，则从该文件加载；
        否则按空状态初始化（attack_methods_dict 为 (1,1) 先验）。
        """
        default_path = load_path if load_path is not None else _get_pe_save_path()
        loaded = load_posterior_evidence(default_path)
        if loaded is not None:
            self.prompt_categories_dict = loaded.prompt_categories_dict
            self.total_success_count = loaded.total_success_count
            self.total_failure_count = loaded.total_failure_count
            self.attack_methods_dict = loaded.attack_methods_dict
        else:
            self.prompt_categories_dict = {}
            self.total_success_count = 0
            self.total_failure_count = 0
            self.attack_methods_dict = {}
            for method_name in attack_methods_list:
                self.attack_methods_dict[method_name] = (1, 1)  # (success_count, failure_count)

    def save(self, path: Optional[str] = None) -> str:
        """保存当前实例到本地（默认工作区根目录），返回保存的绝对路径。"""
        return save_posterior_evidence(self, path)

    def create_category(self, prompt_category_name: str) -> None:
        if prompt_category_name not in self.prompt_categories_dict:
            self.prompt_categories_dict[prompt_category_name] = PromptCategory()
    def update_success(self, prompt_category_name: str, attackmethod_name: str, prompt: str, example1: str, example2: str) -> None:
        self.prompt_categories_dict[prompt_category_name].update_success(attackmethod_name, prompt, example1, example2)
        self.total_success_count += 1
        self.attack_methods_dict[attackmethod_name] = (self.attack_methods_dict[attackmethod_name][0] + 1, self.attack_methods_dict[attackmethod_name][1])
    def update_failure(self, prompt_category_name: str, attackmethod_name: str) -> None:
        self.prompt_categories_dict[prompt_category_name].update_failure(attackmethod_name)
        self.total_failure_count += 1
        self.attack_methods_dict[attackmethod_name] = (self.attack_methods_dict[attackmethod_name][0], self.attack_methods_dict[attackmethod_name][1] + 1)

        
    def select_attack_method(self, prompt_category_name: str, prompt: str, lambda_ =1.0, gamma =1.0) -> str:
        alpha0 = self.total_success_count
        beta0 = self.total_failure_count
        score_dict = {}
        for attackmethod_name in attack_methods_list:
            alpha_a = alpha0 + self.attack_methods_dict[attackmethod_name][0]
            beta_a = beta0 + self.attack_methods_dict[attackmethod_name][1]
            alpha_ac = alpha_a + self.prompt_categories_dict[prompt_category_name].attack_methods_dict[attackmethod_name].success_count
            beta_ac = beta_a + self.prompt_categories_dict[prompt_category_name].attack_methods_dict[attackmethod_name].failure_count
            mu = alpha_ac / (alpha_ac + beta_ac)
            sigma = math.sqrt((alpha_ac * beta_ac) / ((alpha_ac + beta_ac) ** 2 * (alpha_ac + beta_ac + 1)))
            
            bayes_ucb = mu + lambda_ * sigma
            max_cosine_similarity = self.prompt_categories_dict[prompt_category_name].calculate_max_cosine_similarity(prompt, attackmethod_name)
            score = math.log(bayes_ucb) + gamma * max_cosine_similarity
            score_dict[attackmethod_name] = score
        # Softmax 采样：按 score 的 softmax 概率随机选择 attack method
        keys = list(score_dict.keys())
        scores = np.array([score_dict[k] for k in keys])
        scores_stable = scores - np.max(scores)  # 数值稳定
        probs = np.exp(scores_stable) / np.sum(np.exp(scores_stable))
        attack_method = np.random.choice(keys, p=probs)
        try:
            examples = self.prompt_categories_dict[prompt_category_name].attack_methods_dict[attack_method].examples
        except KeyError:
            examples = []
        return attack_method, examples
            