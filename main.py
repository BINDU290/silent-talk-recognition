import torch
import hydra
import cv2
import time
from pipelines.pipeline import InferencePipeline
import numpy as np
from datetime import datetime
from ollama import chat
from pydantic import BaseModel
import keyboard
from concurrent.futures import ThreadPoolExecutor
import os


# pydantic model for the chat output
class BinduOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Bindu:
    def __init__(self):
        self.vsr_model = None

        # flag to toggle recording
        self.recording = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

    def perform_inference(self, video_path):
        output = self.vsr_model(video_path)

        keyboard.write(output)

        # shift left to select the entire output
        cmd = ', '.join(['shift+left'] * len(output))
        keyboard.press_and_release(cmd)

        # get corrected text from chat model
        response = chat(
            model='gemma3:4b',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You are an assistant that helps make corrections to the output of a lipreading model. "
                        "The text you will receive was transcribed using a video-to-text system that attempts to lipread "
                        "the subject speaking in the video, so the text will likely be imperfect.\n\n"
                        "If something seems unusual, assume it was mistranscribed. Do your best to infer the words actually spoken, "
                        "and make changes to the mistranscriptions in your response. Do not add more words or content, just change "
                        "the ones that seem to be out of place (and, therefore, mistranscribed). Do not change even the wording of "
                        "sentences, just individual words that look nonsensical in the context of all of the other words in the sentence.\n\n"
                        "Also, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'. "
                        "The input text in all-caps, although your response should be capitalized correctly and should NOT be in all-caps.\n\n"
                        "Return the corrected text in the format of 'list_of_changes' and 'corrected_text'."
                    )
                },
                {
                    'role': 'user',
                    'content': f"Transcription:\n\n{output}"
                }
            ],
            format=BinduOutput.model_json_schema()
        )

        chat_output = BinduOutput.model_validate_json(response.message.content)

        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        keyboard.write(chat_output.corrected_text + " ")

        return {"output": chat_output.corrected_text, "video_path": video_path}

    def start_webcam(self):
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_frame_time = time.time()
        futures = []
        output_path = ""
        out = None
        frame_count = 0

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if out is not None:
                    out.release()

                for file in os.listdir():
                    if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                        for _ in range(5):
                            try:
                                os.remove(file)
                                break
                            except PermissionError:
                                time.sleep(0.5)
                break

            current_time = time.time()
            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if not ret:
                    continue

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

                if self.recording:
                    if out is None:
                        output_path = f"{self.output_prefix}{time.time_ns() // 1_000_000}.mp4"
                        out = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            self.fps,
                            (frame_width, frame_height),
                            False
                        )
                    out.write(compressed_frame)
                    last_frame_time = current_time
                    cv2.circle(compressed_frame, (frame_width - 20, 20), 10, (0, 0, 0), -1)
                    frame_count += 1

                elif not self.recording and frame_count > 0:
                    if out is not None:
                        out.release()
                        out = None

                    if frame_count >= self.fps * 2:
                        futures.append(self.executor.submit(self.perform_inference, output_path))
                    else:
                        for _ in range(5):
                            try:
                                if os.path.exists(output_path):
                                    os.remove(output_path)
                                break
                            except PermissionError:
                                time.sleep(0.5)

                    output_path = f"{self.output_prefix}{time.time_ns() // 1_000_000}.mp4"
                    out = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        self.fps,
                        (frame_width, frame_height),
                        False
                    )
                    frame_count = 0

                cv2.imshow('Bindu', cv2.flip(compressed_frame, 1))

            for fut in futures[:]:
                if fut.done():
                    result = fut.result()
                    video_path = result["video_path"]
                    for _ in range(5):
                        try:
                            if os.path.exists(video_path):
                                os.remove(video_path)
                            break
                        except PermissionError:
                            time.sleep(0.5)
                    futures.remove(fut)

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    def on_action(self, event):
        if event.event_type == keyboard.KEY_DOWN and event.name == 'alt':
            self.recording = not self.recording


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    bindu = Bindu()
    keyboard.hook(lambda e: bindu.on_action(e))
    bindu.vsr_model = InferencePipeline(
        cfg.config_filename,
        device=torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu"),
        detector=cfg.detector,
        face_track=True
    )
    print("Model loaded successfully!")
    bindu.start_webcam()


if __name__ == '__main__':
    main()