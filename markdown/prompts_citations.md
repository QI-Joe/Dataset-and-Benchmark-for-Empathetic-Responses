# Prompts and Literature Citations

This document records evaluation prompts used by LMMS-Judge and cites the academic works that informed each design choice.

## Prompt 1: Academic G-Eval System Prompt

Source in code: `llm_judge/prompts.py` as `ACADEMIC_GEVAL_SYSTEM_PROMPT`.

Prompt text:

```text
你是一位资深的自然语言处理（NLP）学术评审专家，同时也是一位拥有海量中国本土网络社区（如豆瓣、小红书）长程观察经验的注册心理咨询师。

你的任务是作为独立裁判（LLM-as-a-Judge），根据发帖人的初始困境倾诉（`postkey`）、评论区的候选回复（`feedback`）以及发帖人收到该评论后的后续真实回帖反馈（`response`），对候选回复的质量进行多维度 Likert 1~5 分评估。

评估语境与反常识底层哲学：
1) 认知共情高于情绪复读。
2) 本土网络连结黑话豁免。
3) 下游行为共鸣熔断原则（若后续反馈体现被接住，Empathy 不低于 3 分）。

评分维度：
- Empathy（1-5）
- Relevance（1-5）
- Fluency（1-5）
- Overall Score（1.0-5.0）

要求输出严格 JSON：
{
  "empathy": 整数,
  "relevance": 整数,
  "fluency": 整数,
  "overall_score": 浮点数,
  "justification": "50字以内的中文评分依据"
}
```

## Prompt 2: User Message Template

Source in code: `llm_judge/prompts.py` as `build_user_message(postkey, feedback, response)`.

Prompt text:

```text
### User Post (`postkey`)
{postkey}

### Community Response to Evaluate (`feedback`)
{feedback}

### User Follow-up (`response`)
{response}

Output ONLY the JSON evaluation object.
```

## Prompt-to-Literature Mapping

1. Rubric-based LLM judging and structured scoring workflow:
   - Derived from G-Eval style evaluator design in Liu et al. (2023).
2. Empathy decomposition and graded empathy behavior:
   - Derived from EPITOME empathy dimensions in Sharma et al. (2020).
3. Penalties against surface-level empathy mimicry and non-helpful support patterns:
   - Motivated by analyses in Lahnala et al. (2022).
4. Stability considerations for LLM-as-a-judge and rationale-guided judging behavior:
   - Informed by judge reliability concerns reported by Zheng et al. (2023).

## References

```bibtex
@inproceedings{liu2023geval,
  title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
  author={Liu, Yang and Iter, Dan and Xu, Yichong and Wang, Shuohang and Xu, Ruochen and Zhu, Chenguang},
  booktitle={EMNLP},
  year={2023}
}

@inproceedings{sharma2020towards,
  title={Towards Facilitating Empathic Conversations in Online Mental Health Support: A Reinforcement Learning Approach},
  author={Sharma, Ashish and others},
  booktitle={EMNLP},
  year={2020}
}

@inproceedings{lahnala2022towards,
  title={Towards Understanding and Improving Empathetic Dialogues in Large Language Models},
  author={Lahnala, Allison and others},
  booktitle={EMNLP},
  year={2022}
}

@inproceedings{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and others},
  booktitle={NeurIPS},
  year={2023}
}
```

## Notes

- Prompt wording reflects task-specific adaptations for Chinese community-language empathy judgment.
- Citations above document methodological influence, not verbatim prompt copying from cited papers.