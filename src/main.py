import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import fire
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from env import RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')


class PromptImprovement:
    def __init__(self, openai_api_key: str | None, anthropic_api_key: str | None):
        self.openai_client: OpenAI = OpenAI(api_key=openai_api_key)
        self.anthropic_client: Anthropic = Anthropic(api_key=anthropic_api_key)

    def improve_with_llm(
        self, model: Literal['gpt-4o', 'claude-3.5-sonnet'], prompt: str, num_iter: int
    ) -> dict[str, Any]:
        match model:
            case 'gpt-4o':
                response = self.openai_client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {
                            'role': 'system',
                            'content': 'Claudeが作成したプロンプトです。これよりも質の高いプロンプトを作って下さい。',
                        },
                        {
                            'role': 'user',
                            'content': f'プロンプト: {prompt}',
                        },
                    ],
                )
                improved_prompt = response.choices[0].message.content
            case 'claude-3.5-sonnet':
                response = self.anthropic_client.messages.create(  # type: ignore
                    model='claude-3-5-sonnet-20241022',
                    max_tokens=1024,
                    messages=[
                        {
                            'role': 'user',
                            'content': f'GPT-4が作成したプロンプトです。これよりも質の高いプロンプトを作って下さい。\nプロンプト: {prompt}',
                        },
                    ],
                )
                improved_prompt = response.content[0].text  # type: ignore
        result = {'prompt': prompt, 'improved_prompt': improved_prompt, 'num_iter': num_iter, 'model': model}
        return result

    def save_as_indented_json(
        self,
        data: dict | list,
        path: str | Path,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=parents, exist_ok=exist_ok)
        with path.open(mode='w', encoding='utf-8') as fout:
            json.dump(data, fout, ensure_ascii=False, indent=4, separators=(',', ': '))


def run(initial_prompt: str, iterations: int) -> None:
    current_prompt = initial_prompt
    prompt_improvement = PromptImprovement(openai_api_key=OPENAI_API_KEY, anthropic_api_key=ANTHROPIC_API_KEY)

    logger.info('Starting experiment...')

    for i in range(iterations):
        logger.info(f'\nIteration {i+1}')

        # GPT improvement
        logger.info('Improving with GPT-4o...')
        result = prompt_improvement.improve_with_llm(model='gpt-4o', prompt=current_prompt, num_iter=i + 1)
        current_prompt = result['improved_prompt']
        prompt_improvement.save_as_indented_json(data=result, path=RESULTS_DIR / 'GPT-4o_{i}.json')
        logger.info(f'Iter{i}: Finish GPT-4o turn\n')

        # Claude improvement
        logger.info('Improving with Claude 3.5 Sonnet...')
        result = prompt_improvement.improve_with_llm(model='claude-3.5-sonnet', prompt=current_prompt, num_iter=i + 1)
        prompt_improvement.save_as_indented_json(data=result, path=RESULTS_DIR / 'Claude-3.5-Sonnet_{i}.json')
        current_prompt = result['improved_prompt']
        logger.info(f'Iter{i}: Finish Claude 3.5 Sonnet turn\n')
    logger.info('Finished.')


if __name__ == '__main__':
    fire.Fire(run)
