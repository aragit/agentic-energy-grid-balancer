"""LLM inference engine with pluggable backends."""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str


class BaseLLMEngine(ABC):
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def shutdown(self):
        pass


class ReasoningEngine(BaseLLMEngine):
    def __init__(self, seed: int = 42):
        import random

        self.rng = random.Random(seed)
        self.model_name = "rule-based-reasoner"
        logger.info("[LLM] Using ReasoningEngine")

    def chat_completion(self, messages, temperature=0.7, max_tokens=512):
        start = time.time()
        prompt = messages[0]["content"] if messages else ""
        agent_type = self._extract_field(prompt, "agent_type:")
        market_price = self._extract_float(prompt, "market_price:")
        balance = self._extract_float(prompt, "balance:")
        demand = self._extract_float(prompt, "demand:")

        content = self._generate_strategy(agent_type, market_price, balance, demand)
        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            tokens_in=len(str(messages)),
            tokens_out=len(content.split()),
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def _extract_field(self, text, key):
        idx = text.lower().find(key.lower())
        if idx == -1:
            return ""
        start = idx + len(key)
        end = text.find("\n", start)
        return text[start:end].strip() if end != -1 else text[start:].strip()

    def _extract_float(self, text, key):
        try:
            idx = text.lower().find(key.lower())
            if idx == -1:
                return 0.0
            start = idx + len(key)
            # FIX: Skip currency symbols, whitespace, colons, equals
            while start < len(text) and text[start] in " \t$=:,":
                start += 1
            end = start
            while end < len(text) and (text[end].isdigit() or text[end] in ".-"):
                end += 1
            return float(text[start:end])
        except (ValueError, IndexError):
            return 0.0

    def _generate_strategy(self, agent_type, market_price, balance, demand):
        if "battery" in agent_type.lower():
            # FIX: Buy low, sell high (was backwards)
            if market_price < 40:
                bid = round(market_price * self.rng.uniform(0.9, 0.95), 2)
                action = "charge"  # BUY when cheap
                reasoning = "Market price is low, buying energy to store for later"
            elif market_price > 70:
                bid = round(market_price * self.rng.uniform(1.05, 1.1), 2)
                action = "discharge"  # SELL when expensive
                reasoning = "High prices, selling stored energy for profit"
            else:
                bid = round(market_price * self.rng.uniform(0.95, 1.05), 2)
                action = "hold"
                reasoning = "Moderate prices, holding position"
        elif "solar" in agent_type.lower() or "wind" in agent_type.lower():
            bid = round(market_price * self.rng.uniform(0.85, 0.95), 2)
            action = "sell"
            reasoning = "Renewable generation, selling at competitive price"
        elif "coal" in agent_type.lower():
            min_price = 45 + self.rng.uniform(0, 10)
            bid = max(round(market_price * self.rng.uniform(1.0, 1.2), 2), min_price)
            action = "sell" if market_price > min_price else "ramp_down"
            reasoning = f"Coal baseload, minimum viable price ${min_price}/MWh"
        elif "nuclear" in agent_type.lower():
            bid = round(market_price * self.rng.uniform(0.8, 0.9), 2)
            action = "sell"
            reasoning = "Nuclear must-run generation, accepting market price"
        elif "consumer" in agent_type.lower():
            if market_price > 80:
                bid = round(market_price * self.rng.uniform(0.5, 0.7), 2)
                action = "reduce_demand"
                reasoning = "High prices, reducing non-essential demand"
            else:
                bid = round(market_price * self.rng.uniform(0.9, 1.1), 2)
                action = "maintain"
                reasoning = "Prices acceptable, maintaining demand"
        else:
            bid = round(market_price * self.rng.uniform(0.9, 1.1), 2)
            action = "sell"
            reasoning = "Default strategy, following market"

        return json.dumps(
            {
                "bid_price": bid,
                "output_adjustment": action,
                "carbon_trade": 0.0,
                "reasoning": reasoning,
                "confidence": round(self.rng.uniform(0.7, 0.95), 2),
            }
        )

    def shutdown(self):
        pass


class OllamaEngine(BaseLLMEngine):
    def __init__(self, model="qwen2.5:1.5b", host="http://localhost:11434"):
        self.model = model
        self.host = host
        self.model_name = model
        logger.info(f"[LLM] Using Ollama: {model}")

    def chat_completion(self, messages, temperature=0.7, max_tokens=512):
        import httpx

        start = time.time()
        response = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
        latency_ms = (time.time() - start) * 1000
        return LLMResponse(
            content=content,
            tokens_in=0,
            tokens_out=0,
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def shutdown(self):
        pass


class LLMEngineFactory:
    @staticmethod
    def create(backend=None, ollama_model="qwen2.5:1.5b"):
        backend = backend or os.getenv("LLM_BACKEND", "mock")
        if backend == "ollama":
            try:
                return OllamaEngine(model=ollama_model)
            except Exception as e:
                logger.warning(f"[LLM] Ollama failed ({e}), falling back to mock")
                return ReasoningEngine()
        return ReasoningEngine()
