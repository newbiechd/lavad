import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.data.video_record import VideoRecord
from src.utils.path_utils import find_unprocessed_videos


class LLMAnomalyScorer:
    def __init__(
        self,
        root_path,
        annotationfile_path,
        batch_size,
        frame_interval,
        summary_prompt,
        context_prompt,
        format_prompt,
        output_scores_dir,
        output_summary_dir,
        captions_dir,
        temperature,
        top_p,
        max_seq_len,
        max_gen_len,
    ):
        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.summary_prompt = summary_prompt
        self.context_prompt = context_prompt
        self.format_prompt = format_prompt
        self.output_scores_dir = output_scores_dir
        self.output_summary_dir = output_summary_dir
        self.captions_dir = captions_dir
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len

        self.generator = LLM(
            model="/data/changhd_data/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            tensor_parallel_size=2,
            max_num_seqs=1,
            max_model_len=self.max_seq_len,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_gen_len,
        )

    def _prepare_prompts(self, captions, batch_frame_idxs, is_summary):
        if is_summary:
            system = self.context_prompt + " " + self.format_prompt
            batch_clip_caption = [f"{captions[str(idx)]}." for idx in batch_frame_idxs]
        else:
            system = self.summary_prompt
            batch_clip_caption = [
                "\n ".join([
                    captions[str(idx)][str(frame_idx)] for frame_idx in captions[str(idx)]
                ])
                for idx in batch_frame_idxs
            ]

        prompts = [
            f"<|im_start|>user\n{system}\n{clip}<|im_end|>\n<|im_start|>assistant<think>\n\n</think>"
            for clip in batch_clip_caption
        ]
        return prompts

    def _generate_temporal_summaries(self, video, video_captions):
        temporal_summaries = {}

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video.path}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), video.num_frames
            )
            batch_frame_idxs = range(batch_start_frame, batch_end_frame, self.frame_interval)

            prompts = self._prepare_prompts(video_captions, batch_frame_idxs, is_summary=False)

            results = self.generator.generate(prompts, self.sampling_params)

            for result, clip_frame_idx in zip(results, batch_frame_idxs):
                # print(f"Response for frame {clip_frame_idx}: {result}")
                response_text = result.outputs[0].text.strip()
                # print(f"Response text: {response_text}")
                if "</think>" in response_text:
                    summary = response_text.split("</think>")[-1].strip()
                else:
                    summary = response_text.strip()
                temporal_summaries[str(clip_frame_idx)] = summary

        return temporal_summaries

    def _parse_score(self, response):
        pattern = r"\[(\d+(?:\.\d+)?)\]"
        match = re.search(pattern, response)
        score = float(match.group(1)) if match else -1
        return score

    def _interpolate_unmatched_scores(self, scores):
        valid_scores = [(idx, score) for idx, score in scores.items() if score != -1]
        video_scores = np.interp(list(scores.keys()), *zip(*valid_scores))
        return dict(zip(scores.keys(), video_scores))

    def _score_temporal_summaries(self, video, temporal_summaries):
        video_scores = {}

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video.path}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), video.num_frames
            )
            batch_frame_idxs = range(batch_start_frame, batch_end_frame, self.frame_interval)

            prompts = self._prepare_prompts(temporal_summaries, batch_frame_idxs, is_summary=True)

            results = self.generator.generate(prompts, self.sampling_params)

            for result, frame_idx in zip(results, batch_frame_idxs):
                print(f"Response for frame {frame_idx}: {result}")
                response_text = result.outputs[0].text.strip()
                print(f"Response text: {response_text}")
                if "</think>" in response_text:
                    response_score = response_text.split("</think>")[-1].strip()
                else:
                    response_score = response_text.strip()
                score = self._parse_score(response_score)
                video_scores[str(frame_idx)] = score

        video_scores = self._interpolate_unmatched_scores(video_scores)
        return video_scores

    def process_video(self, video, score_summary):
        video_name = Path(video.path).name

        if not score_summary:
            video_caption_path = Path(self.captions_dir) / f"{video_name}.json"
            with open(video_caption_path) as f:
                video_captions = json.load(f)

            output_path = Path(self.output_summary_dir) / f"{video_name}.json"

            if not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporal_summaries = self._generate_temporal_summaries(video, video_captions)
                with open(output_path, "w") as f:
                    json.dump(temporal_summaries, f, indent=4)
        else:
            temporal_summaries_path = Path(self.output_summary_dir) / f"{video_name}.json"
            with open(temporal_summaries_path) as f:
                temporal_summaries = json.load(f)

            video_scores = self._score_temporal_summaries(video, temporal_summaries)

            output_path = Path(self.output_scores_dir) / f"{video_name}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(video_scores, f, indent=4)


def run(
    root_path,
    annotationfile_path,
    batch_size,
    frame_interval,
    summary_prompt,
    context_prompt,
    format_prompt,
    output_scores_dir,
    output_summary_dir,
    captions_dir,
    temperature,
    top_p,
    max_seq_len,
    max_gen_len,
    resume,
    pathname,
    num_jobs,
    job_index,
    score_summary,
):
    if score_summary:
        output_scores_dir = Path(output_scores_dir)
        output_scores_dir.mkdir(parents=True, exist_ok=True)
        with open(output_scores_dir / "context_prompt.txt", "w") as f:
            f.write(context_prompt)
        with open(output_scores_dir / "format_prompt.txt", "w") as f:
            f.write(format_prompt)
    else:
        output_summary_dir = Path(output_summary_dir)
        output_summary_dir.mkdir(parents=True, exist_ok=True)
        with open(output_summary_dir / "summary_prompt.txt", "w") as f:
            f.write(summary_prompt)

    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]
    video_list = list(np.array_split(video_list, num_jobs)[job_index])
    if resume:
        video_list = find_unprocessed_videos(
            video_list, output_scores_dir if score_summary else output_summary_dir, pathname
        )

    llm_anomaly_scorer = LLMAnomalyScorer(
        root_path=root_path,
        annotationfile_path=annotationfile_path,
        batch_size=batch_size,
        frame_interval=frame_interval,
        summary_prompt=summary_prompt,
        context_prompt=context_prompt,
        format_prompt=format_prompt,
        output_scores_dir=output_scores_dir,
        output_summary_dir=output_summary_dir,
        captions_dir=captions_dir,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
        max_gen_len=max_gen_len,
    )

    for video in video_list:
        llm_anomaly_scorer.process_video(video, score_summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--summary_prompt", type=str)
    parser.add_argument("--context_prompt", type=str)
    parser.add_argument("--format_prompt", type=str)
    parser.add_argument("--output_scores_dir", type=str)
    parser.add_argument("--output_summary_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_gen_len", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pathname", type=str, default="*.json")
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--score_summary", action="store_true")

    args = parser.parse_args()

    if args.score_summary:
        if not (args.context_prompt and args.format_prompt and args.output_scores_dir):
            parser.error("--context_prompt, --format_prompt, and --output_scores_dir are required for scoring the temporal summaries.")
    else:
        if not (args.captions_dir and args.summary_prompt):
            parser.error("--captions_dir and --summary_prompt are required for generating the temporal summaries.")

    return args


if __name__ == "__main__":
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        summary_prompt=args.summary_prompt,
        context_prompt=args.context_prompt,
        format_prompt=args.format_prompt,
        output_scores_dir=args.output_scores_dir,
        output_summary_dir=args.output_summary_dir,
        captions_dir=args.captions_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        resume=args.resume,
        pathname=args.pathname,
        num_jobs=args.num_jobs,
        job_index=args.job_index,
        score_summary=args.score_summary,
    )
