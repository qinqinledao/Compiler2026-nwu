#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolutionary Algorithm Main Logic Module - Focused on Generational Analysis & Exploitation
"""

import random
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from config import EAConfig
from genetic_operators import GeneticOperators, Individual
from executor import TritonExecutor, EvaluationResult


class EvolutionaryAlgorithm:

    def __init__(self, 
                 genetic_ops: GeneticOperators,
                 executor: TritonExecutor,
                 config: EAConfig):
 
        self.genetic_ops = genetic_ops
        self.executor = executor
        self.config = config

        # Population state
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None

    def initialize_population(self, seed_codes: List[str]) -> None:
        print(f"[EA] Initializing population, target size: {self.config.population_size}")

        # Step 1: 塞入基线种子选手
        for i, code in enumerate(seed_codes):
            if i >= self.config.population_size:
                break
            self.population.append(Individual(code=code, generation=0))

        # Step 2: 宏观调配 —— 指挥武器库进行前置瓶颈分析
        required_variants = self.config.population_size - len(self.population)
        print(f"[EA] 正在对种子代码进行开局多路线（共 {required_variants} 条）策略发掘...")
    
        strategies = self.genetic_ops.analyze_kernel(seed_codes[0], required_variants)

        # Step 3: 根据定制赛道方向进行精准变异填充
        strategy_idx = 0
        while len(self.population) < self.config.population_size:
            seed_code = random.choice(seed_codes)
            temp_ind = Individual(code=seed_code, generation=0)
            
            hint = strategies[strategy_idx] if strategy_idx < len(strategies) else None
            new_ind = self.genetic_ops.mutate(temp_ind, strategy_hint=hint)
            self.population.append(new_ind)
            strategy_idx += 1

        # Step 4: 交付评估
        self._evaluate_population()
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        print(f"[EA] Initial best fitness: {self.best_individual.fitness:.4f}")

    def select_parents(self) -> Tuple[Individual, Individual]:
        fitnesses = [ind.fitness for ind in self.population]
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            return random.sample(self.population, 2)

        probabilities = [f / total_fitness for f in fitnesses]
        selected_indices = np.random.choice(
            len(self.population),
            size=2,
            replace=False,
            p=probabilities
        )
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def evolve_generation(self) -> None:
        new_population = []

        # 🌟 核心升级：繁衍开盘前，对当前代的所有选手进行宏观横向大复盘
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        print(f"[EA] 📊 正在激活全局诊断大师，审计提取当前代的演进红黑榜报告...")
        generation_report = self.genetic_ops.analyze_population_trends(sorted_pop)
        print(f"[EA] 🧠 代际趋势报告提炼完毕，已成功广播至所有的繁衍通道。")

        # Step 1: 精英保留机制
        elite_count = int(self.config.elite_ratio * self.config.population_size)
        elites = sorted_pop[:elite_count]
        new_population.extend(elites)
        print(f"[EA] Preserving {len(elites)} elite individuals")

        # Step 2: 优胜劣汰繁衍后代
        while len(new_population) < self.config.population_size:
            parent1, parent2 = self.select_parents()

            if random.random() < self.config.crossover_rate:
                # 将全局诊断报告无缝挂载传入交叉
                child = self.genetic_ops.crossover(parent1, parent2, global_report=generation_report)
            else:
                child = parent1 if parent1.fitness > parent2.fitness else parent2

            if random.random() < self.config.mutation_rate:
                # 将全局诊断报告无缝挂载传入突变
                child = self.genetic_ops.mutate(child, global_report=generation_report)

            new_population.append(child)

        self.population = new_population
        self._evaluate_population()

        self.generation += 1
        current_best = max(self.population, key=lambda x: x.fitness)

        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
            print(f"[EA] Found better individual, fitness: {current_best.fitness:.4f}")

        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        print(f"[EA] Gen {self.generation}: Best={current_best.fitness:.4f}, Avg={avg_fitness:.4f}")

    def _evaluate_population(self) -> None:
        """带有【底层日志捕获】与【平滑梯度】的评估流（不触发自动修复）"""
        import uuid
        for ind in self.population:
            if not ind.id or ind.id.startswith('hash_') or len(ind.id) != 8:
                ind.id = uuid.uuid4().hex[:8]

            if ind.fitness == 0 and not ind.metadata.get('evaluated', False):
                result: EvaluationResult = self.executor.evaluate(ind.code)
                
                # 仅静默捕获底层报错记录到元数据，不启动 fix 抢救
                real_error = result.error
                if not result.success:
                    log_path = Path(self.executor.performance_dir) / self.executor.kernel_name / "get_prof.log"
                    if log_path.exists():
                        try:
                            with open(log_path, 'r', encoding='utf-8') as f:
                                log_content = f.read()
                                if log_content.strip():
                                    real_error = log_content[-1500:]
                        except Exception:
                            pass
                    print(f"       [⚠️ 真机崩溃] 算子编译或运行失败，错误已被记录至元数据。")

                # 平滑评分补丁：慢于基线也不打 0 分，完美瞒天过海，保留进化的火种
                final_fitness = result.fitness
                if result.success and final_fitness == 0.0 and result.execution_time > 0:
                    final_fitness = self.executor.baseline_time / result.execution_time

                ind.fitness = final_fitness
                ind.metadata.update({
                    'evaluated': True,
                    'success': result.success,
                    'speedup': result.speedup,
                    'execution_time': result.execution_time,
                    'error': real_error
                })

    def run(self, seed_codes: List[str]) -> Individual:
        self.initialize_population(seed_codes)
        for gen in range(self.config.max_generations):
            print(f"\n[EA] ===== Generation {gen + 1}/{self.config.max_generations} =====")
            self.evolve_generation()

            if self.best_individual and self.best_individual.fitness >= 1.9:
                print(f"[EA] Reached performance limit, early stop (fitness={self.best_individual.fitness:.4f})")
                break

        print(f"\n[EA] Evolution complete, best fitness: {self.best_individual.fitness:.4f}")
        return self.best_individual