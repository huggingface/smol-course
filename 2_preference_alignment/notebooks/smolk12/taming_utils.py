
import asyncio
import aiohttp
import json
import logging
import time
from typing import List, Dict, Any, Optional

class OpenAIBatchProcessor:
    """
    Handles batch processing of OpenAI API requests with rate limiting and error handling.
    """
    def __init__(
        self,
        api_key: str,
        max_requests_per_minute: int = 1500,  # 50% of the default 3000 limit
        max_tokens_per_minute: int = 125000,  # 50% of the default 250000 limit
        max_retries: int = 5,
        cooldown_period: int = 15  # seconds to wait after rate limit error
    ):
        self.api_key = api_key
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_retries = max_retries
        self.cooldown_period = cooldown_period
        
        # Initialize rate limiting trackers
        self.available_requests = max_requests_per_minute
        self.available_tokens = max_tokens_per_minute
        self.last_update_time = time.time()
        
        # Initialize status tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limit_errors = 0

    async def process_batch(
        self,
        requests: List[Dict[str, Any]],
        save_filepath: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of requests to the OpenAI API.
        
        Args:
            requests: List of request dictionaries
            save_filepath: Optional path to save results
        
        Returns:
            List of responses/errors for each request
        """
        results = []
        retry_queue = asyncio.Queue()
        
        async with aiohttp.ClientSession() as session:
            # Create initial tasks
            tasks = [
                self._process_single_request(session, request, retry_queue)
                for request in requests
            ]
            
            # Process initial requests
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(completed_results)
            
            # Process retry queue
            while not retry_queue.empty():
                retry_request = await retry_queue.get()
                result = await self._process_single_request(
                    session, 
                    retry_request["request"],
                    retry_queue,
                    retry_count=retry_request["retries"]
                )
                results.append(result)
        
        # Save results if filepath provided
        if save_filepath:
            self._save_results(results, save_filepath)
            
        return results

    async def _process_single_request(
        self,
        session: aiohttp.ClientSession,
        request: Dict[str, Any],
        retry_queue: asyncio.Queue,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Process a single API request with rate limiting and error handling."""
        
        # Update rate limit capacity
        await self._update_rate_limits()
        
        # Check if we have capacity
        if not self._has_capacity(request):
            await asyncio.sleep(1)  # Wait for capacity
            return await self._process_single_request(session, request, retry_queue, retry_count)
        
        # Consume capacity
        self._consume_capacity(request)
        
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request
            ) as response:
                if response.status == 429:  # Rate limit error
                    self.rate_limit_errors += 1
                    await asyncio.sleep(self.cooldown_period)
                    if retry_count < self.max_retries:
                        await retry_queue.put({
                            "request": request,
                            "retries": retry_count + 1
                        })
                    return {"error": "rate_limit_exceeded", "request": request}
                
                result = await response.json()
                self.successful_requests += 1
                return result
                
        except Exception as e:
            if retry_count < self.max_retries:
                await retry_queue.put({
                    "request": request,
                    "retries": retry_count + 1
                })
            self.failed_requests += 1
            return {"error": str(e), "request": request}

    async def _update_rate_limits(self):
        """Update available rate limit capacity based on time elapsed."""
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        
        # Update available capacity
        self.available_requests = min(
            self.available_requests + (self.max_requests_per_minute * time_elapsed / 60.0),
            self.max_requests_per_minute
        )
        self.available_tokens = min(
            self.available_tokens + (self.max_tokens_per_minute * time_elapsed / 60.0),
            self.max_tokens_per_minute
        )
        
        self.last_update_time = current_time

    def _has_capacity(self, request: Dict[str, Any]) -> bool:
        """Check if we have enough capacity to process request."""
        estimated_tokens = self._estimate_tokens(request)
        return (self.available_requests >= 1 and 
                self.available_tokens >= estimated_tokens)

    def _consume_capacity(self, request: Dict[str, Any]):
        """Consume capacity for a request."""
        estimated_tokens = self._estimate_tokens(request)
        self.available_requests -= 1
        self.available_tokens -= estimated_tokens

    def _estimate_tokens(self, request: Dict[str, Any]) -> int:
        """Estimate token usage for a request. Implement more sophisticated estimation if needed."""
        # Simple estimation - can be improved based on your needs
        return 1000  # Conservative estimate

    def _save_results(self, results: List[Dict[str, Any]], filepath: str):
        """Save results to a JSONL file."""
        with open(filepath, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')



from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import csv

class ParallelEvaluator:
    def __init__(self, eval):
        self.eval = eval
        
    def process_chunk(self, prompts_chunk):
        return self.eval.generate_responses(prompts_chunk)
        
    def evaluate(self, prompts, n_parts, save_to_csv: str | None = None):
        # Split prompts into n chunks
        chunks = np.array_split(prompts, n_parts)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_parts) as executor:
            results = list(executor.map(self.process_chunk, chunks))
        
        results_df = pd.concat(results, ignore_index=True)
        if save_to_csv:
            results_df.to_csv(save_to_csv, quoting=csv.QUOTE_ALL, index=False)

        return results_df
    



from openai import OpenAI

MAX_NEW_TOKENS = 1500

class ChosenResponseGenerator:
    """Generates high-quality chosen responses using OpenAI's GPT-4."""
    
    def __init__(self, policy_content: str, model: str = "gpt-4o-mini"):
        """Initialize OpenAI client with input policy.
        
        Args:
            input_policy: String describing the policy responses should follow
        """
        self.client = OpenAI()
        self.policy = policy_content
        self.model = model
            
    def generate_responses(self, prompts: list[str], save_to_csv: str | None = None) -> pd.DataFrame:
        """Generate chosen responses for a list of prompts.
        
        Args:
            prompts: List of user prompts
            
        Returns:
            DataFrame with prompts and generated chosen responses
        """
        ALIGNED_SUFFIX = f"I'm sorry, but I can't help with that since it violates {COMPANY_NAME}'s policy."
        system_prompt = f"""
        ROLE: You are a helpful assistant.
        TASK: Generate a response to the user prompt
        REQUIREMENTS: 
        - Response must comply with the provided POLICY
        - If chat is not aligned with POLICY, you MUST return your response with the following suffix: "{ALIGNED_SUFFIX}" and explain why it violates the POLICY.
        - Keep the response short
        POLICY: {self.policy}
        """
        responses = []
        
        
        for prompt in prompts:
            # Remove enclosing quotes if present
            prompt = prompt.strip('"')
            
            try:
                # Generate response using GPT-4
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                response = completion.choices[0].message.content
                responses.append(response)
                print(f"Generated {len(responses)}/{len(prompts)} responses...")
                
                # Add delay to respect rate limits
                #time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating response for prompt: {prompt}")
                print(f"Error: {str(e)}")
                responses.append("")
                
        results_df = pd.DataFrame({
            "user_prompts": prompts,
            "model_responses": responses
        })
        
        if save_to_csv:
            results_df.to_csv(save_to_csv, quoting=csv.QUOTE_ALL, index=False)
            
        return results_df