"""
Test-Time Scaling (TTS) Script with Beam Search and Self-Rewarding
This script implements beam search for each of the four reasoning steps:
1. PROBLEM: Problem understanding and intention analysis
2. CAPTION: Image description and visual analysis
3. REASONING: Logical reasoning and step-by-step analysis
4. OUTPUT: Final answer and conclusion

For each step, it generates multiple candidates and uses self-rewarding to select the best ones.
"""

import json
import torch
import logging
import argparse
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
from datetime import datetime
import threading
from openai import OpenAI
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import existing modules
from utils import read_json, write_json, apply_chat_template, load_actor_model
from orm import evaluate, extract_answer
from config import MCTSBaseConfig


@dataclass
class TTSConfig(MCTSBaseConfig):
    """Extended configuration for TTS with beam search"""
    
    # Beam search parameters
    beam_width: int = field(
        default=5, metadata={"help": "Number of candidates to keep at each step"}
    )
    num_candidates: int = field(
        default=10, metadata={"help": "Number of initial candidates to generate at each step"}
    )
    
    # Best-of-N parameters
    final_candidates: int = field(
        default=8, metadata={"help": "Number of complete responses to generate for final selection"}
    )
    
    # Step-specific generation parameters
    step_max_tokens: Dict[str, int] = field(
        default_factory=lambda: {
            "PROBLEM": 512,
            "CAPTION": 512, 
            "REASONING": 1024,
            "OUTPUT": 512
        }, metadata={"help": "Max tokens for each step"}
    )
    
    # Self-rewarding parameters
    use_step_rewards: bool = field(
        default=True, metadata={"help": "Use step-wise self-rewarding during beam search"}
    )
    use_final_rewards: bool = field(
        default=True, metadata={"help": "Use final self-rewarding for best-of-N selection"}
    )
    
    min_diversity_threshold: float = field(
        default=1.0, metadata={"help": "Minimum similarity threshold (set high to disable)"}
    )


class BeamSearchNode:
    """Represents a single beam in the search"""
    
    def __init__(self, trajectory: List[str], score: float = 0.0, step: int = 0):
        self.trajectory = trajectory.copy()
        self.score = score
        self.step = step
        self.step_scores = []  # Individual step scores
        
    def add_step(self, content: str, step_score: float = 0.0):
        """Add a new step to the trajectory"""
        self.trajectory.append(content)
        self.step_scores.append(step_score)
        self.score += step_score
        self.step += 1
        
    def copy(self):
        """Create a copy of this node"""
        new_node = BeamSearchNode(self.trajectory, self.score, self.step)
        new_node.step_scores = self.step_scores.copy()
        return new_node


class TTSGenerator:
    """Main TTS generator with beam search and self-rewarding"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.steps = ["PROBLEM", "CAPTION", "REASONING", "OUTPUT"]
        self.step_patterns = {
            "PROBLEM": ("<|PROBLEM|>", "<|/PROBLEM|>"),
            "CAPTION": ("<|CAPTION|>", "<|/CAPTION|>"),
            "REASONING": ("<|REASONING|>", "<|/REASONING|>"),
            "OUTPUT": ("<|OUTPUT|>", "<|/OUTPUT|>")
        }
        
        # Initialize model and tokenizer
        self._init_model()

    def _normalize_image_paths(self, image_paths) -> List[str]:
        """Normalize image_paths input to a list of file paths."""
        if image_paths is None:
            return []
        if isinstance(image_paths, str):
            return [image_paths]
        if isinstance(image_paths, (list, tuple)):
            # Filter out empty/None and ensure strings
            return [p for p in image_paths if isinstance(p, str) and p]
        # Fallback: convert to string
        return [str(image_paths)]
        
    def _init_model(self):
        """Initialize the model and tokenizer"""
        if self.config.generate_mode == "local":
            self.actor_model, self.actor_tokenizer = load_actor_model(self.config.actor_model_dir)
            self.actor_model.eval()
        else:
            openai_api_key = "EMPTY"
            openai_api_base = self.config.server_url
            self.actor_model = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            self.actor_tokenizer = AutoTokenizer.from_pretrained(self.config.actor_model_dir)
    
    def _generate_step_candidates(
        self, 
        prompt: str, 
        trajectory: List[str], 
        step: str, 
        num_candidates: int,
        image_paths
    ) -> List[str]:
        """Generate multiple candidates for a specific step"""
        candidates = []
        
        # Generate candidates using the current trajectory as-is
        # The step marker handling is done in the VLLM generation
        for _ in range(num_candidates):
            try:
                response = self._generate_single_response(
                    prompt, trajectory, step, image_paths
                )
                if response:
                    # Ensure the response has proper closing tag
                    processed_response = self._process_step_response(response, step)
                    if processed_response and processed_response not in candidates:
                        candidates.append(processed_response)
            except Exception as e:
                logging.warning(f"Error generating candidate for step {step}: {e}")
                continue
                
        return candidates[:num_candidates]  # Ensure we don't exceed requested number
    
    def _process_step_response(self, response: str, step: str) -> str:
        """Process generated response to ensure proper closing tags"""
        step_start, step_end = self.step_patterns[step]
        
        # If response doesn't contain the closing tag, add it
        if step_end not in response:
            # Find where to add the closing tag
            if step_start in response:
                # Split at the opening tag and ensure content ends with closing tag
                parts = response.split(step_start, 1)
                if len(parts) == 2:
                    content = parts[1].strip()
                    # Add closing tag if not present
                    if not content.endswith(step_end):
                        response = parts[0] + step_start + content + "\n" + step_end
                    else:
                        response = parts[0] + step_start + content
            else:
                # Response doesn't have opening tag, add both
                response = step_start + "\n" + response.strip() + "\n" + step_end
        
        return response
    
    def _generate_single_response(
        self, 
        prompt: str, 
        trajectory: List[str], 
        step: str,
        image_paths
    ) -> str:
        """Generate a single response using the model"""
        terminators = [self.actor_tokenizer.eos_token_id]
        
        # Add step-specific stop token
        step_start, step_end = self.step_patterns[step]
        step_stop_token = step_end  # e.g., "<|/PROBLEM|>"
        
        # For VLLM, we'll pass the string directly
        # For local generation, we need to convert to token ID
        if self.config.generate_mode == "local":
            try:
                step_stop_token_id = self.actor_tokenizer.convert_tokens_to_ids(step_stop_token)
                terminators.append(step_stop_token_id)
            except:
                logging.warning(f"Could not convert step stop token to ID: {step_stop_token}")
        
        max_tokens = self.config.step_max_tokens.get(step, self.config.max_tokens)
        
        if self.config.generate_mode == "local":
            return self._generate_local(prompt, trajectory, terminators, max_tokens)
        else:
            return self._generate_vllm(prompt, trajectory, terminators, max_tokens, step, image_paths, step_stop_token)
    
    def _generate_local(
        self, 
        prompt: str, 
        trajectory: List[str], 
        terminators: List[int], 
        max_tokens: int
    ) -> str:
        """Generate using local model"""
        full_prompt = apply_chat_template(prompt, trajectory, self.actor_tokenizer)
        input_ids = self.actor_tokenizer.encode(
            full_prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.actor_model.device)
        
        with torch.no_grad():
            outputs = self.actor_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                pad_token_id=self.actor_tokenizer.eos_token_id,
            ).cpu()
        
        response = outputs[0][input_ids.shape[-1]:]
        return self.actor_tokenizer.decode(response, skip_special_tokens=False)
    
    def _generate_vllm(
        self, 
        prompt: str, 
        trajectory: List[str], 
        terminators: List[int], 
        max_tokens: int,
        step: str,
        image_paths,
        step_stop_token: Optional[str] = None
    ) -> str:
        """Generate using VLLM server"""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Add the main question/prompt
        user_message = {"role": "user", "content": prompt}
        user_message["content"] = [
            {"type": "image_url", "image_url": {"url": f"file://{img}"}}
            for img in image_paths
        ] + [{"type": "text", "text": prompt}]
        
        messages.append(user_message)
        
        # Add trajectory and prepare for next step (following original STAIR pattern)
        prefix_str = ""
        
        if trajectory:
            trajectory_content = "\n\n".join(trajectory)
            
            # Determine next step based on current step
            if step == "CAPTION" and not trajectory_content.endswith("<|/PROBLEM|>"):
                trajectory_content += "\n\n<|CAPTION|>\n"
                prefix_str = "<|CAPTION|>\n"
            elif step == "REASONING" and not trajectory_content.endswith("<|/CAPTION|>"):
                trajectory_content += "\n\n<|REASONING|>\n"
                prefix_str = "<|REASONING|>\n"
            elif step == "OUTPUT" and not trajectory_content.endswith("<|/REASONING|>"):
                trajectory_content += "\n\n<|OUTPUT|>\n"
                prefix_str = "<|OUTPUT|>\n"
            
            messages.append({"role": "assistant", "content": trajectory_content})
        elif step == "PROBLEM":
            # For the first step (PROBLEM), start with the step marker
            prefix_str = "<|PROBLEM|>\n"
        
        # Use step-specific stop token if provided
        stop_strings = []
        if step_stop_token:
            stop_strings.append(step_stop_token)
        
        if trajectory:
            request_body = {
                "model": "actor",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stop": stop_strings,
                "extra_body": {
                    "top_k": self.config.top_k,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                    "include_stop_str_in_output": True,
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                }
            }
        else:
            request_body = {
                "model": "actor",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stop": stop_strings,
                "extra_body": {
                    "top_k": self.config.top_k,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                    "include_stop_str_in_output": True,
                }
            }
        
        completion = self.actor_model.chat.completions.create(**request_body)
        response_content = completion.choices[0].message.content
        
        # Add prefix if needed (for continuation)
        if prefix_str and not response_content.startswith(prefix_str):
            response_content = prefix_str + response_content
            
        return response_content
    
    def _evaluate_step_candidate(
        self, 
        prompt: str, 
        candidate: str, 
        step_index: int
    ) -> float:
        """Evaluate a candidate for a specific step"""
        if not self.config.use_step_rewards:
            return 0.0
        
        try:
            depth = step_index + 1

            score = evaluate(
                self.config.mode, prompt, candidate, depth
            )
            
            return score if score is not None else 0.0
                
        except Exception as e:
            logging.warning(f"Error evaluating step candidate: {e}")
            return 0.0
     
    
    def _beam_search_step(
        self, 
        prompt: str, 
        current_beams: List[BeamSearchNode], 
        step_index: int,
        image_paths: List[str] = None
    ) -> List[BeamSearchNode]:
        """Perform beam search for a single step"""
        step = self.steps[step_index]
        new_beams = []
        
        for beam in current_beams:
            # Generate candidates for this beam
            candidates = self._generate_step_candidates(
                prompt, beam.trajectory, step, self.config.num_candidates, image_paths
            )            
            # Score each candidate
            scored_candidates = []
            existing_contents = [beam.trajectory[-1] if beam.trajectory else ""]
            
            for candidate in candidates:
                
                # Extract the step content
                try:
                    step_content = extract_answer(candidate, step_index + 1)
                except Exception as e:
                    continue
                
                if step_content == "Wrong Format":
                    continue
                
                # Evaluate the candidate
                try:
                    step_score = self._evaluate_step_candidate(
                        prompt, candidate, step_index
                    )
                except Exception as e:
                    step_score = 0.0
                
                scored_candidates.append((candidate, step_score))
                existing_contents.append(step_content)
            
            # Select top candidates for this beam
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored_candidates[:self.config.beam_width]
            
            # Create new beams
            for candidate, score in top_candidates:
                new_beam = beam.copy()
                new_beam.add_step(candidate, score)
                new_beams.append(new_beam)
        
        # Keep only top beams across all expansions
        new_beams.sort(key=lambda x: x.score, reverse=True)
        return new_beams[:self.config.beam_width]
    
    def generate_with_beam_search(
        self, 
        prompt: str, 
        image_paths
    ) -> List[BeamSearchNode]:
        """Generate complete responses using beam search"""
        # Normalize image paths
        image_paths = self._normalize_image_paths(image_paths)

        # Initialize with empty beam
        current_beams = [BeamSearchNode([], 0.0, 0)]
        
        # Perform beam search for each step
        for step_index in range(len(self.steps)):
            logging.info(f"Performing beam search for step {step_index + 1}: {self.steps[step_index]}")
            current_beams = self._beam_search_step(
                prompt, current_beams, step_index, image_paths
            )
            
            if not current_beams:
                logging.warning(f"No valid beams after step {step_index + 1}")
                break
        
        return current_beams
    
    def generate_complete_response(
        self, 
        prompt: str, 
        image_paths
    ) -> BeamSearchNode:
        """Generate a single complete response without step-wise selection"""
        image_paths = self._normalize_image_paths(image_paths)

        current_beam = BeamSearchNode([], 0.0, 0)
        
        # Generate each step sequentially without beam search
        for step_index in range(len(self.steps)):
            step = self.steps[step_index]
            
            try:
                # Generate single candidate for this step
                candidates = self._generate_step_candidates(
                    prompt, current_beam.trajectory, step, 1, image_paths
                )
                
                if candidates:
                    # Take the first (and only) candidate without scoring
                    candidate = candidates[0]
                    current_beam.add_step(candidate, 0.0)  # No step score
                else:
                    logging.warning(f"No candidates generated for step {step}")
                    break
                    
            except Exception as e:
                logging.warning(f"Error generating step {step}: {e}")
                break
        
        return current_beam
    
    def generate_best_of_n(
        self, 
        prompt: str, 
        image_paths
    ) -> Dict[str, Any]:
        """Generate multiple complete responses and select the best one"""
        image_paths = self._normalize_image_paths(image_paths)

        all_responses = []
        
        # Check if this is a pure Best-of-N scenario (beam_width=1 and no step rewards)
        if (self.config.beam_width == 1 and self.config.num_candidates == 1 and 
            not self.config.use_step_rewards):
            # Generate N complete responses without intermediate selection
            for i in range(self.config.final_candidates):
                response = self.generate_complete_response(prompt, image_paths)
                if response.trajectory:  # Only add if we have a complete response
                    all_responses.append(response)
        else:
            # Use beam search approach
            for i in range(self.config.final_candidates // self.config.beam_width + 1):
                beams = self.generate_with_beam_search(prompt, image_paths)
                all_responses.extend(beams)
                
                if len(all_responses) >= self.config.final_candidates:
                    break
            
            # Limit to requested number of candidates
            all_responses = all_responses[:self.config.final_candidates]
        
        if not all_responses:
            return {
                "best_response": {"trajectory": [], "score": 0.0, "step_scores": []},
                "all_responses": [],
                "selection_method": "none"
            }
        
        # If final rewards are enabled, re-evaluate complete responses
        if self.config.use_final_rewards:
            for response in all_responses:
                if len(response.trajectory) == len(self.steps):
                    # Evaluate the complete response
                    complete_answer = "\n\n".join(response.trajectory)
                    try:
                        score = evaluate(
                            self.config.mode, prompt, complete_answer, 
                            len(self.steps)
                        )
                        
                        # Use helpful_score for all cases (no distinction between safety and helpfulness)
                        score = score if score is not None else 0.0
                        # Update score (could weight step vs final scores)
                        response.score = 0.7 * response.score + 0.3 * score
                        
                    except Exception as e:
                        logging.warning(f"Error in final evaluation: {e}")
        
        # Select best response
        all_responses.sort(key=lambda x: x.score, reverse=True)
        best_response = all_responses[0]
        
        return {
            "best_response": {
                "trajectory": best_response.trajectory,
                "score": best_response.score,
                "step_scores": best_response.step_scores
            },
            "all_responses": [
                {
                    "trajectory": resp.trajectory,
                    "score": resp.score,
                    "step_scores": resp.step_scores
                } for resp in all_responses
            ],
            "selection_method": "best_of_n" if (self.config.beam_width == 1 and not self.config.use_step_rewards) else "beam_search_with_self_reward"
        }


def process_single_prompt(
    prompt_data: Dict[str, Any], 
    generator: TTSGenerator
) -> Dict[str, Any]:
    """Process a single prompt with TTS"""
    question = prompt_data["question"]
    ground_truth = prompt_data.get("response", None)
    question_type = prompt_data.get("type", "helpfulness")
    image_path = prompt_data.get("image", None)
    # Normalize image_path to a list of paths (or empty list)
    if isinstance(image_path, str):
        image_paths = [image_path]
    elif isinstance(image_path, list):
        image_paths = image_path
    else:
        image_paths = []
    logging.info(f"Processing question: {question[:100]}...")
    
    try:
        result = generator.generate_best_of_n(
            question, image_paths
        )
        
        # Add original prompt data
        result.update({
            "question": question,
            "ground_truth": ground_truth,
            "question_type": question_type,
            "image_path": image_paths,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing prompt: {e}")
        return {
            "question": question,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def thread_function(prompts_data: List[Dict], config: TTSConfig, worker_order: int):
    """Worker thread function"""
    logging.info(f"THREAD {worker_order} BEGIN")
    
    try:
        generator = TTSGenerator(config)
        output_content = []
        
        for prompt_index, prompt in enumerate(tqdm(prompts_data, desc=f"Worker {worker_order}")):
                
            result = process_single_prompt(prompt, generator)
            output_content.append(result)
        
        # Save results
        output_file = os.path.join(config.output_path, f"tts_results_{worker_order}.json")
        write_json(output_file, output_content)
        
    except Exception as e:
        logging.error(f"Error in thread {worker_order}: {e}")
    
    logging.info(f"THREAD {worker_order} END")


def parse_args():
    parser = argparse.ArgumentParser(description="TTS with Beam Search and Self-Rewarding")
    parser.add_argument(
        '--custom_cfg', 
        type=str, 
        default="../config/tts_config.yaml",
        help="Path to custom configuration file"
    )
    parser.add_argument(
        '--beam_width', 
        type=int, 
        default=5,
        help="Beam width for search"
    )
    parser.add_argument(
        '--num_candidates', 
        type=int, 
        default=10,
        help="Number of candidates to generate at each step"
    )
    parser.add_argument(
        '--final_candidates', 
        type=int, 
        default=1,
        help="Number of final candidates for best-of-N selection"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.structured(TTSConfig)
    if args.custom_cfg and os.path.exists(args.custom_cfg):
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    
    # Override with command line arguments
    if args.beam_width:
        config.beam_width = args.beam_width
    if args.num_candidates:
        config.num_candidates = args.num_candidates
    if args.final_candidates:
        config.final_candidates = args.final_candidates
    
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    
    # Setup logging
    os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"TTS CONFIG: {config}")
    
    # Load prompts
    prompts_data = read_json(config.train_prompt_path)
    logging.info(f"PROMPT DATA LOADED: {len(prompts_data)} prompts")
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    # Run processing
    threads = []
    for i in range(config.worker_num):
        start_idx = min(i * config.worker_prompt_num, len(prompts_data))
        end_idx = min((i + 1) * config.worker_prompt_num, len(prompts_data))
        prompts_data_for_worker = prompts_data[start_idx:end_idx]
        
        thread = threading.Thread(
            target=thread_function, 
            args=(prompts_data_for_worker, config, i)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    logging.info("All workers completed!")


if __name__ == '__main__':
    main()
