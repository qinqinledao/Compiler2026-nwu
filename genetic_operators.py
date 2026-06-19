#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Operators Module - Full Indentation Aligned with Generational Diagnostics
"""

import random
import re
import ast
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import uuid


@dataclass
class Individual:
    code: str
    fitness: float = 0.0
    generation: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: dict = field(default_factory=dict)
    model_used: str = "unknown"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Individual):
            return False
        return self.id == other.id


class GeneticOperators:

    def __init__(self, llm, config):
        self.llm = llm
        self.config = config

    def _escape_code_for_prompt(self, code: str) -> str:
        return code.replace("{", "{{").replace("}", "}}")

    def analyze_population_trends(self, sorted_population: List[Individual]) -> str:
        """【高级分析节点】横向对比当前种群的优等生与落后生，提炼出代际优化红黑榜"""
        if len(sorted_population) < 2:
            return "No historical trends available yet."

        # 抓取当前种群最顶尖的 2 个优等生
        top_individuals = sorted_population[:2]
        # 抓取当前种群表现最差的 2 个落后生（包括编译失败或极慢的）
        bottom_individuals = sorted_population[-2:]

        prompt_parts = [
            "You are a Triton Compiler Optimization Supervisor for Huawei Ascend NPU.",
            "Analyze the following performance gap trends within the current evolutionary generation.",
            "",
            "=== SUCCESSFUL WINNERS (Fastest in current generation) ===",
        ]
        
        for idx, ind in enumerate(top_individuals):
            p_time = ind.metadata.get('execution_time', 0.0)
            prompt_parts.append(f"Winner #{idx+1} (Fitness: {ind.fitness:.4f}, Latency: {p_time:.2f} us):")
            prompt_parts.append(self._escape_code_for_prompt(ind.code))
            prompt_parts.append("")

        prompt_parts.append("=== FAILURE/SLOWER LOSERS (Worst in current generation) ===")
        for idx, ind in enumerate(bottom_individuals):
            err_msg = ind.metadata.get('error') or f"{ind.metadata.get('execution_time', 0.0)} us"
            prompt_parts.append(f"Loser #{idx+1} (Fitness: {ind.fitness:.4f}, Error/Latency: {err_msg}):")
            prompt_parts.append(self._escape_code_for_prompt(ind.code))
            prompt_parts.append("")

        prompt_parts.extend([
            "CRITICAL TASK FOR GENERATION UPGRADE:",
            "Compare the winner code patterns against the loser code patterns to discover the underlying hardware reasons on Huawei Ascend NPU.",
            "Provide a brief, high-impact summary covering:",
            "1. WHAT WORKS: What specific coding pattern, tile size, or indexing logic made the winners fast? (MUST BE INHERITED)",
            "2. WHAT FAILS: What specific configuration, dimension, or syntax rule caused the losers to slow down or crash? (MUST BE BANNED)",
            "Output ONLY the bulleted analysis report. No conversational filler, no code blocks."
        ])

        report = self.llm.generate(
            "\n".join(prompt_parts),
            system_msg="You are a senior compiler diagnostics tool. Output only the generation summary report.",
            purpose='analysis'
        )
        return report

    def crossover(self, parent1: Individual, parent2: Individual, global_report: str = None) -> Individual:
        """【升级功能】融合全种种群趋势报告与显式性能反哺的自适应交叉算子"""
        selected_model = random.choice([
            parent1.model_used if parent1.model_used != "unknown" else self.llm.current_model,
            parent2.model_used if parent2.model_used != "unknown" else self.llm.current_model
        ])

        if selected_model != self.llm.current_model and selected_model in self.config.llm_models:
            self.llm.switch_model(selected_model)

        p1_time = parent1.metadata.get('execution_time', 0.0)
        p2_time = parent2.metadata.get('execution_time', 0.0)
        
        trend_context = f"\n--- CURRENT GENERATION GLOBAL INSIGHTS (CRITICAL TRENDS) ---\n{global_report}\n-----------------------------------------------------------\n" if global_report else ""
         
        prompt_parts = [
            "You are a Triton kernel optimization expert exclusively for Huawei Ascend NPU.",
            "Combine the best optimization strategies from both parent kernels to create a faster offspring kernel.",
            trend_context,
            "",
            f"Parent 1 (PERFORMANCE RANK: BETTER if {parent1.fitness} > {parent2.fitness} else WORSE):",
            f"  - Fitness Score: {parent1.fitness:.4f} (Speedup Ratio indicator)",
            f"  - Actual Measured Latency: {p1_time:.2f} us",
            f"  - Code Base:",
            self._escape_code_for_prompt(parent1.code),
            "",
            f"Parent 2 (PERFORMANCE RANK: BETTER if {parent2.fitness} > {parent1.fitness} else WORSE):",
            f"  - Fitness Score: {parent2.fitness:.4f}",
            f"  - Actual Measured Latency: {p2_time:.2f} us",
            f"  - Code Base:",
            self._escape_code_for_prompt(parent2.code),
            "",
            "CRITICAL CRITERIA FOR CONSTRUCTING OFFSPRING:",
            "1. Strictly align with the rules in the CURRENT GENERATION GLOBAL INSIGHTS if provided.",
            "2. Identify why the BETTER parent runs faster. Keep its superior structure.",
            "3. Gently inject any clever masking or localized enhancement from the other parent if safe.",
            "4. DYNAMIC WORKLOAD ADAPTATION: You MUST write adaptive logic within the 'act_quant' wrapper to size BLOCK_M dynamically based on runtime input size.",
            "5. Keep function signatures compatible, BLOCK_SIZE must be a multiple of 16 (use 16, 32, 64, or 128), and num_warps conservative.",
            "6. Output ONLY valid Python Triton code, no markdown blocks, no explanations.",
            "",
            "Generate the optimized hybrid offspring kernel:"
        ]
        prompt = "\n".join(prompt_parts)

        new_code = self.llm.generate(
            prompt, 
            system_msg="Generate only valid Python Triton code optimized for Ascend NPU.",
            purpose='crossover',
            parents_fitness=f"{parent1.fitness:.3f}, {parent2.fitness:.3f}",
            generation=max(parent1.generation, parent2.generation) + 1,
            model=self.llm.current_model
        )
        return Individual(
            code=self._clean_code(new_code),
            generation=max(parent1.generation, parent2.generation) + 1,
            metadata={'parents': [parent1.id, parent2.id], 'operation': 'crossover_with_report'},
            model_used=self.llm.current_model
        )

    def analyze_kernel(self, base_code: str, num_directions: int) -> List[str]:
        """【保留功能】利用 purpose='analysis' 开局发掘多赛道初始战略"""
        prompt = "\n".join([
            "You are an expert Triton kernel developer for Huawei Ascend NPU.",
            "Analyze the following baseline kernel and propose distinct, high-impact optimization strategies.",
            f"Each strategy must be a specific concrete direction (e.g., 'Change tiling shape to favor small dimensions').",
            f"Provide exactly {num_directions} distinct strategies.",
            "",
            f"Baseline Code:\n{self._escape_code_for_prompt(base_code)}",
            "",
            f"Output ONLY a clean numbered list containing exactly {num_directions} text strings (one per line), no explanations, no markdown blocks."
        ])
        
        raw_analysis = self.llm.generate(
            prompt, 
            system_msg="You are a hardware-aware Triton compiler expert. Output only strategies.",
            purpose='analysis'
        )
        
        strategies = [line.strip() for line in raw_analysis.split('\n') if line.strip()]
        strategies = [re.sub(r'^\s*\d+\.\s*', '', s) for s in strategies]
        
        if len(strategies) >= num_directions:
            return strategies[:num_directions]
        return [f"Hyper-parameter tuning sweep variant {i}" for i in range(num_directions)]

    def mutate(self, individual: Individual, strategy_hint: str = None, global_report: str = None) -> Individual:
        """【升级功能】带全种种群大局观趋势与方向指导（Hint）的自适应突变算子"""
        mutation_types = ['param_tuning', 'strategy_change', 'local_rewrite']
        code_mutation_type = random.choice(mutation_types)
        
        if strategy_hint:
            code_mutation_type = 'strategy_change'

        if random.random() < 0.2 and len(self.config.llm_models) > 1:
            other_models = [m for m in self.config.llm_models if m != self.llm.current_model]
            if other_models:
                new_model = random.choice(other_models)
                self.llm.switch_model(new_model)

        hint_announcement = f"\nCRITICAL STRATEGIC DIRECTION: You MUST implement: {strategy_hint}\n" if strategy_hint else ""
        trend_context = f"\n--- CURRENT GENERATION GLOBAL INSIGHTS (DO NOT REPEAT LOSER MISTAKES) ---\n{global_report}\n--------------------------------------------------------------------------\n" if global_report else ""

        prompt_parts = [
            "You are an expert in Huawei Ascend NPU Triton kernel optimization.",
            f"Perform {code_mutation_type} mutation on the following Triton kernel.",
            trend_context,
            hint_announcement,
            f"Original Kernel (Fitness: {individual.fitness:.3f}):",
            self._escape_code_for_prompt(individual.code),
            "",
            "CRITICAL RULES FOR ASCEND NPU (MUST FOLLOW):",
            "1. Read the GLOBAL INSIGHTS carefully: Avoid any patterns flagged in 'WHAT FAILS', and creatively expand on 'WHAT WORKS'.",
            "2. DYNAMIC WORKLOAD ADAPTATION: Rewrite 'act_quant' wrapper to adaptively set BLOCK_M based on dimension M.",
            "3. Ascend Cube architecture strictly requires BLOCK_SIZE to be a multiple of 16 (use 16, 32, 64, or 128).",
            "4. NEVER change 'num_stages' unless perfectly synchronized with wrapper logic.",
            "5. Output ONLY valid Python Triton code, no markdown blocks, no explanations.",
            "",
            "Generate the mutated adaptive kernel:"
        ]
        prompt = "\n".join(prompt_parts)

        new_code = self.llm.generate(
            prompt,
            system_msg="You are a Triton optimization expert. Generate only valid Python Triton code.",
            purpose='mutate',
            parent_fitness=individual.fitness,
            mutation_type=code_mutation_type,
            generation=individual.generation + 1,
            model=self.llm.current_model
        )
        return Individual(
            code=self._clean_code(new_code),
            generation=individual.generation + 1,
            metadata={'parent': individual.id, 'operation': 'mutation_with_report', 'hint': strategy_hint},
            model_used=self.llm.current_model
        )

    def _clean_code(self, raw_code: str) -> str:
        code = re.sub(r'```python\s*', '', raw_code, flags=re.IGNORECASE)
        code = re.sub(r'```\s*', '', code)
        code = re.sub(r'^\s*\d+\.\s*', '', code, flags=re.MULTILINE)
        return code.strip()

    def _validate_code(self, code: str) -> Tuple[bool, str]:
        if 'import triton' not in code or 'def ' not in code:
            return False, "Missing critical structural elements"
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {str(e)}"
        return True, "Valid"

    def model_crossover(self, model1: str, model2: str) -> str:
        return random.choice([model1, model2])