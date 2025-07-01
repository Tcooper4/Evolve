"""
MultimodalAgent: Visual reasoning agent for trading analytics.
- Generates plots (Matplotlib/Plotly)
- Passes images to vision models (OpenAI GPT-4V or BLIP)
- Produces natural language insights on equity curve, drawdown, and performance
"""

import io
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from base_agent_interface import BaseAgent, AgentConfig, AgentResult
from prompt_templates import format_template

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

class MultimodalAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="MultimodalAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={}
            )
        super().__init__(config)
        
        # Extract config from custom_config or use defaults
        custom_config = config.custom_config or {}
        self.openai_api_key = custom_config.get('openai_api_key') or (openai.api_key if openai else None)
        self.use_blip = custom_config.get('use_blip', False) and BlipProcessor and BlipForConditionalGeneration and Image
        
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
        if self.use_blip:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the multimodal analysis logic.
        Args:
            **kwargs: equity, returns, action, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get('action', 'analyze_equity_curve')
            
            if action == 'analyze_equity_curve':
                equity = kwargs.get('equity')
                
                if equity is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: equity"
                    )
                
                result = self.analyze_equity_curve(equity)
                return AgentResult(success=True, data={
                    "analysis_result": result['result'],
                    "message": result['message']
                })
                
            elif action == 'analyze_drawdown':
                equity = kwargs.get('equity')
                
                if equity is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: equity"
                    )
                
                result = self.analyze_drawdown(equity)
                return AgentResult(success=True, data={
                    "analysis_result": result['result'],
                    "message": result['message']
                })
                
            elif action == 'analyze_performance':
                returns = kwargs.get('returns')
                
                if returns is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: returns"
                    )
                
                result = self.analyze_performance(returns)
                return AgentResult(success=True, data={
                    "analysis_result": result['result'],
                    "message": result['message']
                })
                
            elif action == 'plot_equity_curve':
                equity = kwargs.get('equity')
                title = kwargs.get('title', "Equity Curve")
                
                if equity is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: equity"
                    )
                
                image_bytes = self.plot_equity_curve(equity, title)
                return AgentResult(success=True, data={
                    "image_bytes_length": len(image_bytes),
                    "title": title
                })
                
            elif action == 'plot_drawdown':
                equity = kwargs.get('equity')
                title = kwargs.get('title', "Drawdown")
                
                if equity is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: equity"
                    )
                
                image_bytes = self.plot_drawdown(equity, title)
                return AgentResult(success=True, data={
                    "image_bytes_length": len(image_bytes),
                    "title": title
                })
                
            elif action == 'plot_performance':
                returns = kwargs.get('returns')
                title = kwargs.get('title', "Strategy Performance")
                
                if returns is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: returns"
                    )
                
                image_bytes = self.plot_performance(returns, title)
                return AgentResult(success=True, data={
                    "image_bytes_length": len(image_bytes),
                    "title": title
                })
                
            elif action == 'vision_insight':
                image_bytes = kwargs.get('image_bytes')
                custom_prompt = kwargs.get('prompt')
                
                if image_bytes is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: image_bytes"
                    )
                
                # Use centralized template if no custom prompt provided
                if custom_prompt is None:
                    prompt = format_template("vision_chart_analysis", vision_prompt="Describe the trading chart and its key features.")
                else:
                    prompt = custom_prompt
                
                insight = self.vision_insight(image_bytes, prompt)
                return AgentResult(success=True, data={
                    "insight": insight,
                    "prompt": prompt
                })
                
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
                
        except Exception as e:
            return self.handle_error(e)

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

    def vision_insight(self, image_bytes: bytes, prompt: str = None) -> str:
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
        prompt = format_template("vision_equity_curve")
        return {'success': True, 'result': self.vision_insight(img, prompt=prompt), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def analyze_drawdown(self, equity: List[float]) -> str:
        img = self.plot_drawdown(equity)
        prompt = format_template("vision_drawdown_analysis")
        return {'success': True, 'result': self.vision_insight(img, prompt=prompt), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def analyze_performance(self, returns: List[float]) -> str:
        img = self.plot_performance(returns)
        prompt = format_template("vision_performance_analysis")
        return {'success': True, 'result': self.vision_insight(img, prompt=prompt), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}