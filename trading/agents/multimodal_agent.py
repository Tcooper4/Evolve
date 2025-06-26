"""
MultimodalAgent: Visual reasoning agent for trading analytics.
- Generates plots (Matplotlib/Plotly)
- Passes images to vision models (OpenAI GPT-4V or BLIP)
- Produces natural language insights on equity curve, drawdown, and performance
"""

import io
import logging
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objs as go
except ImportError:
    go = None

try:
    import openai
except ImportError:
    openai = None

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    Image = None

logger = logging.getLogger(__name__)

class MultimodalAgent:
    def __init__(self, openai_api_key: Optional[str] = None, use_blip: bool = False):
        self.openai_api_key = openai_api_key or (openai.api_key if openai else None)
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
        self.use_blip = use_blip and BlipProcessor and BlipForConditionalGeneration and Image
        if self.use_blip:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def plot_equity_curve(self, equity: List[float], title: str = "Equity Curve") -> bytes:
        plt.figure(figsize=(8, 4))
        plt.plot(equity, label="Equity")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.read()

    def plot_drawdown(self, equity: List[float], title: str = "Drawdown") -> bytes:
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        plt.figure(figsize=(8, 4))
        plt.plot(drawdown, color='red', label="Drawdown")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Drawdown")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.read()

    def plot_performance(self, returns: List[float], title: str = "Strategy Performance") -> bytes:
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(returns)), returns, color='blue', label="Returns")
        plt.title(title)
        plt.xlabel("Period")
        plt.ylabel("Return")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.read()

    def vision_insight(self, image_bytes: bytes, prompt: str = "Describe the trading chart and its key features.") -> str:
        """Pass image to a vision model and get a natural language insight."""
        if openai and self.openai_api_key:
            # OpenAI GPT-4V API
            import base64
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}]}
                ],
                max_tokens=300
            )
            return response.choices[0].message['content'].strip()
        elif self.use_blip:
            # BLIP model
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.blip_processor(image, prompt, return_tensors="pt")
            out = self.blip_model.generate(**inputs)
            return self.blip_processor.decode(out[0], skip_special_tokens=True)
        else:
            return "[Vision model not available]"

    def analyze_equity_curve(self, equity: List[float]) -> str:
        img = self.plot_equity_curve(equity)
        return self.vision_insight(img, prompt="Describe the equity curve shape, trends, and any notable features for a quant trading engineer.")

    def analyze_drawdown(self, equity: List[float]) -> str:
        img = self.plot_drawdown(equity)
        return self.vision_insight(img, prompt="Describe drawdown spikes and risk periods in this trading equity curve.")

    def analyze_performance(self, returns: List[float]) -> str:
        img = self.plot_performance(returns)
        return self.vision_insight(img, prompt="Describe the strategy's performance over time and any patterns in the returns.") 