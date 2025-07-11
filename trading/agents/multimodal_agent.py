"""
MultimodalAgent: Visual reasoning agent for trading analytics.
- Generates plots (Matplotlib/Plotly)
- Passes images to vision models (OpenAI GPT-4V or BLIP)
- Produces natural language insights on equity curve, drawdown, and performance
"""

import io
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from .base_agent_interface import BaseAgent, AgentConfig, AgentResult
from .prompt_templates import format_template
from dataclasses import dataclass
import base64
import json
import os
from pathlib import Path

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

try:
    import speech_recognition as sr
    import librosa
except ImportError:
    sr = None
    librosa = None

try:
    import PyPDF2
    import docx
    import pandas as pd
except ImportError:
    PyPDF2 = None
    docx = None

logger = logging.getLogger(__name__)

@dataclass
class MultimodalRequest:
    """Request for multimodal processing."""
    text_input: Optional[str] = None
    image_input: Optional[bytes] = None
    audio_input: Optional[bytes] = None
    document_input: Optional[bytes] = None
    data_input: Optional[Dict[str, Any]] = None
    processing_type: str = "analysis"
    output_format: str = "text"
    context: Optional[Dict[str, Any]] = None

@dataclass
class MultimodalResult:
    """Result of multimodal processing."""
    success: bool
    output: Any
    confidence: float
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class ImageHandler:
    """Handler for image processing and analysis."""
    
    def __init__(self, openai_api_key: Optional[str] = None, use_blip: bool = False):
        self.openai_api_key = openai_api_key
        self.use_blip = use_blip and BlipProcessor and BlipForConditionalGeneration and Image
        
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        if self.use_blip:
            try:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("BLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BLIP model: {e}")
                self.use_blip = False
    
    def analyze_image(self, image_bytes: bytes, prompt: str = None) -> Dict[str, Any]:
        """
        Analyze image using available vision models.
        
        Args:
            image_bytes: Image data as bytes
            prompt: Optional prompt for analysis
            
        Returns:
            Analysis result dictionary
        """
        try:
            results = {}
            
            # Try OpenAI GPT-4V if available
            if openai and self.openai_api_key:
                openai_result = self._analyze_with_openai(image_bytes, prompt)
                results['openai'] = openai_result
            
            # Try BLIP if available
            if self.use_blip:
                blip_result = self._analyze_with_blip(image_bytes, prompt)
                results['blip'] = blip_result
            
            # Combine results
            combined_result = self._combine_vision_results(results)
            
            return {
                'success': True,
                'results': results,
                'combined_analysis': combined_result,
                'models_used': list(results.keys())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': {}
            }
    
    def _analyze_with_openai(self, image_bytes: bytes, prompt: str = None) -> Dict[str, Any]:
        """Analyze image using OpenAI GPT-4V."""
        try:
            if not openai or not self.openai_api_key:
                return {'error': 'OpenAI not available'}
            
            # Convert image to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Default prompt for trading charts
            default_prompt = """
            Analyze this trading chart image and provide insights on:
            1. Overall trend direction
            2. Key support/resistance levels
            3. Volume patterns
            4. Technical indicators visible
            5. Potential trading opportunities or risks
            """
            
            analysis_prompt = prompt or default_prompt
            
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return {
                'analysis': response.choices[0].message.content,
                'model': 'gpt-4-vision-preview',
                'confidence': 0.9  # High confidence for GPT-4V
            }
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_with_blip(self, image_bytes: bytes, prompt: str = None) -> Dict[str, Any]:
        """Analyze image using BLIP model."""
        try:
            if not self.use_blip:
                return {'error': 'BLIP not available'}
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Process image
            inputs = self.blip_processor(image, prompt or "Describe this trading chart", return_tensors="pt")
            
            # Generate caption
            out = self.blip_model.generate(**inputs)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return {
                'analysis': caption,
                'model': 'blip',
                'confidence': 0.7  # Medium confidence for BLIP
            }
            
        except Exception as e:
            logger.error(f"BLIP analysis failed: {e}")
            return {'error': str(e)}
    
    def _combine_vision_results(self, results: Dict[str, Any]) -> str:
        """Combine results from multiple vision models."""
        try:
            valid_results = []
            
            for model_name, result in results.items():
                if 'error' not in result and 'analysis' in result:
                    valid_results.append(result['analysis'])
            
            if not valid_results:
                return "No valid analysis results available"
            
            # For now, return the first valid result
            # In the future, could implement more sophisticated combination logic
            return valid_results[0]
            
        except Exception as e:
            logger.error(f"Error combining vision results: {e}")
            return "Error combining analysis results"

class AudioHandler:
    """Handler for audio processing and transcription."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer() if sr else None
        self.use_librosa = librosa is not None
    
    def transcribe_audio(self, audio_bytes: bytes, language: str = 'en-US') -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio_bytes: Audio data as bytes
            language: Language code for transcription
            
        Returns:
            Transcription result dictionary
        """
        try:
            if not self.recognizer:
                return {'error': 'Speech recognition not available'}
            
            # Convert bytes to audio data
            audio_data = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
            
            # Transcribe using Google Speech Recognition
            text = self.recognizer.recognize_google(audio_data, language=language)
            
            return {
                'success': True,
                'transcription': text,
                'language': language,
                'confidence': 0.8
            }
            
        except sr.UnknownValueError:
            return {
                'success': False,
                'error': 'Speech not recognized',
                'confidence': 0.0
            }
        except sr.RequestError as e:
            return {
                'success': False,
                'error': f'Speech recognition service error: {e}',
                'confidence': 0.0
            }
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def analyze_audio_features(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze audio features for sentiment or other characteristics.
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Audio analysis result dictionary
        """
        try:
            if not self.use_librosa:
                return {'error': 'Librosa not available for audio analysis'}
            
            # Convert bytes to numpy array
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes))
            
            # Extract features
            features = {
                'duration': librosa.get_duration(y=audio_array, sr=sr),
                'tempo': librosa.beat.tempo(y=audio_array, sr=sr)[0],
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=sr)),
                'mfcc': np.mean(librosa.feature.mfcc(y=audio_array, sr=sr), axis=1).tolist()
            }
            
            return {
                'success': True,
                'features': features,
                'sample_rate': sr
            }
            
        except Exception as e:
            logger.error(f"Audio feature analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class DocumentHandler:
    """Handler for document parsing and analysis."""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.txt': self._parse_text,
            '.csv': self._parse_csv,
            '.json': self._parse_json
        }
    
    def parse_document(self, document_bytes: bytes, filename: str = None) -> Dict[str, Any]:
        """
        Parse document content based on file format.
        
        Args:
            document_bytes: Document data as bytes
            filename: Optional filename for format detection
            
        Returns:
            Parsed document result dictionary
        """
        try:
            # Determine file format
            if filename:
                file_ext = Path(filename).suffix.lower()
            else:
                # Try to detect format from content
                file_ext = self._detect_file_format(document_bytes)
            
            if file_ext not in self.supported_formats:
                return {
                    'success': False,
                    'error': f'Unsupported file format: {file_ext}'
                }
            
            # Parse document
            parser_func = self.supported_formats[file_ext]
            content = parser_func(document_bytes)
            
            return {
                'success': True,
                'content': content,
                'format': file_ext,
                'filename': filename
            }
            
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _detect_file_format(self, document_bytes: bytes) -> str:
        """Detect file format from content."""
        try:
            # Check for PDF magic number
            if document_bytes.startswith(b'%PDF'):
                return '.pdf'
            
            # Check for ZIP (DOCX is a ZIP file)
            if document_bytes.startswith(b'PK'):
                return '.docx'
            
            # Check for JSON
            try:
                json.loads(document_bytes.decode('utf-8'))
                return '.json'
            except:
                pass
            
            # Check for CSV
            try:
                content = document_bytes.decode('utf-8')
                if ',' in content and '\n' in content:
                    return '.csv'
            except:
                pass
            
            # Default to text
            return '.txt'
            
        except Exception as e:
            logger.error(f"Error detecting file format: {e}")
            return '.txt'
    
    def _parse_pdf(self, document_bytes: bytes) -> str:
        """Parse PDF document."""
        if not PyPDF2:
            raise ImportError("PyPDF2 not available")
        
        pdf_file = io.BytesIO(document_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    def _parse_docx(self, document_bytes: bytes) -> str:
        """Parse DOCX document."""
        if not docx:
            raise ImportError("python-docx not available")
        
        doc = docx.Document(io.BytesIO(document_bytes))
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text.strip()
    
    def _parse_text(self, document_bytes: bytes) -> str:
        """Parse text document."""
        return document_bytes.decode('utf-8', errors='ignore')
    
    def _parse_csv(self, document_bytes: bytes) -> Dict[str, Any]:
        """Parse CSV document."""
        if not pd:
            raise ImportError("pandas not available")
        
        content = document_bytes.decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        
        return {
            'data': df.to_dict('records'),
            'columns': df.columns.tolist(),
            'shape': df.shape
        }
    
    def _parse_json(self, document_bytes: bytes) -> Dict[str, Any]:
        """Parse JSON document."""
        content = document_bytes.decode('utf-8')
        return json.loads(content)

class MultimodalAgent(BaseAgent):
    """Agent for processing multimodal inputs (text, images, audio, data)."""
    
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
        openai_api_key = custom_config.get('openai_api_key') or (openai.api_key if openai else None)
        use_blip = custom_config.get('use_blip', False)
        
        # Initialize handlers
        self.image_handler = ImageHandler(openai_api_key, use_blip)
        self.audio_handler = AudioHandler()
        self.document_handler = DocumentHandler()

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
            action = kwargs.get('action', 'process_multimodal')
            
            if action == 'process_multimodal':
                return await self._process_multimodal_input(kwargs)
            elif action == 'analyze_equity_curve':
                equity = kwargs.get('equity')
                if equity is None:
                    return AgentResult(success=False, error_message="Missing required parameter: equity")
                result = self.analyze_equity_curve(equity)
                return AgentResult(success=True, data={"analysis_result": result['result'], "message": result['message']})
            elif action == 'analyze_drawdown':
                equity = kwargs.get('equity')
                if equity is None:
                    return AgentResult(success=False, error_message="Missing required parameter: equity")
                result = self.analyze_drawdown(equity)
                return AgentResult(success=True, data={"analysis_result": result['result'], "message": result['message']})
            elif action == 'analyze_performance':
                returns = kwargs.get('returns')
                if returns is None:
                    return AgentResult(success=False, error_message="Missing required parameter: returns")
                result = self.analyze_performance(returns)
                return AgentResult(success=True, data={"analysis_result": result['result'], "message": result['message']})
            elif action == 'plot_equity_curve':
                equity = kwargs.get('equity')
                title = kwargs.get('title', "Equity Curve")
                if equity is None:
                    return AgentResult(success=False, error_message="Missing required parameter: equity")
                image_bytes = self.plot_equity_curve(equity, title)
                return AgentResult(success=True, data={"image_bytes_length": len(image_bytes), "title": title})
            elif action == 'plot_drawdown':
                equity = kwargs.get('equity')
                title = kwargs.get('title', "Drawdown")
                if equity is None:
                    return AgentResult(success=False, error_message="Missing required parameter: equity")
                image_bytes = self.plot_drawdown(equity, title)
                return AgentResult(success=True, data={"image_bytes_length": len(image_bytes), "title": title})
            elif action == 'plot_performance':
                returns = kwargs.get('returns')
                title = kwargs.get('title', "Strategy Performance")
                if returns is None:
                    return AgentResult(success=False, error_message="Missing required parameter: returns")
                image_bytes = self.plot_performance(returns, title)
                return AgentResult(success=True, data={"image_bytes_length": len(image_bytes), "title": title})
            elif action == 'vision_insight':
                image_bytes = kwargs.get('image_bytes')
                prompt = kwargs.get('prompt')
                if image_bytes is None:
                    return AgentResult(success=False, error_message="Missing required parameter: image_bytes")
                result = self.image_handler.analyze_image(image_bytes, prompt)
                return AgentResult(success=result['success'], data=result)
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error in multimodal agent execution: {e}")
            return AgentResult(success=False, error_message=str(e))
    
    async def _process_multimodal_input(self, kwargs: Dict[str, Any]) -> AgentResult:
        """Process multimodal input using appropriate handlers."""
        try:
            # Extract inputs
            text_input = kwargs.get('text_input')
            image_input = kwargs.get('image_input') or kwargs.get('image_bytes')
            audio_input = kwargs.get('audio_input') or kwargs.get('audio_bytes')
            document_input = kwargs.get('document_input') or kwargs.get('document_bytes')
            
            results = {}
            
            # Process image if provided
            if image_input:
                image_result = self.image_handler.analyze_image(image_input, kwargs.get('image_prompt'))
                results['image_analysis'] = image_result
            
            # Process audio if provided
            if audio_input:
                audio_result = self.audio_handler.transcribe_audio(audio_input, kwargs.get('language', 'en-US'))
                results['audio_transcription'] = audio_result
                
                # Also analyze audio features
                audio_features = self.audio_handler.analyze_audio_features(audio_input)
                results['audio_features'] = audio_features
            
            # Process document if provided
            if document_input:
                filename = kwargs.get('filename')
                document_result = self.document_handler.parse_document(document_input, filename)
                results['document_parsing'] = document_result
            
            # Combine results
            combined_analysis = self._combine_multimodal_results(results, text_input)
            
            return AgentResult(
                success=True,
                data={
                    'results': results,
                    'combined_analysis': combined_analysis,
                    'input_types': self._get_input_types(kwargs)
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing multimodal input: {e}")
            return AgentResult(success=False, error_message=str(e))
    
    def _combine_multimodal_results(self, results: Dict[str, Any], text_input: Optional[str] = None) -> str:
        """Combine results from multiple modalities."""
        try:
            analysis_parts = []
            
            # Add text input if provided
            if text_input:
                analysis_parts.append(f"Text Input: {text_input}")
            
            # Add image analysis
            if 'image_analysis' in results and results['image_analysis']['success']:
                analysis_parts.append(f"Image Analysis: {results['image_analysis']['combined_analysis']}")
            
            # Add audio transcription
            if 'audio_transcription' in results and results['audio_transcription']['success']:
                analysis_parts.append(f"Audio Transcription: {results['audio_transcription']['transcription']}")
            
            # Add document content
            if 'document_parsing' in results and results['document_parsing']['success']:
                content = results['document_parsing']['content']
                if isinstance(content, str):
                    analysis_parts.append(f"Document Content: {content[:500]}...")
                else:
                    analysis_parts.append(f"Document Data: {str(content)[:500]}...")
            
            if not analysis_parts:
                return "No valid input data to analyze"
            
            return "\n\n".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error combining multimodal results: {e}")
            return "Error combining analysis results"
    
    def _get_input_types(self, kwargs: Dict[str, Any]) -> List[str]:
        """Get list of input types provided."""
        input_types = []
        
        if kwargs.get('text_input'):
            input_types.append('text')
        if kwargs.get('image_input') or kwargs.get('image_bytes'):
            input_types.append('image')
        if kwargs.get('audio_input') or kwargs.get('audio_bytes'):
            input_types.append('audio')
        if kwargs.get('document_input') or kwargs.get('document_bytes'):
            input_types.append('document')
        
        return input_types