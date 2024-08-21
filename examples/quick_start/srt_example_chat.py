"""
Usage:
python3 srt_example_chat.py
"""

import sglang as sgl


@sgl.function
def multi_turn_question(s, question_1, question_2):
    print(f"1 examples/quick_start/srt_example_chat.py question_1:{question_1} and question_2:{question_2} and type(s):{type(s)}")
    
    s += sgl.user(question_1)
    print(f"2 examples/quick_start/srt_example_chat.py, s:{s}:type(s):{type(s)}")
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    print(f"3 examples/quick_start/srt_example_chat.py, s:{s}")
    s += sgl.user(question_2)
    print(f"4 examples/quick_start/srt_example_chat.py, s:{s}")
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))
    print(f"5 examples/quick_start/srt_example_chat.py, s:{s}")

def single():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])


def stream():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
        stream=True,
    )

    for out in state.text_iter():
        print(out, end="", flush=True)
    print()


def batch():
    states = multi_turn_question.run_batch(
        [
            {
                "question_1": "What is the capital of the United States?",
                "question_2": "List two local attractions.",
            },
            {
                "question_1": "What is the capital of France?",
                "question_2": "What is the population of this city?",
            },
        ]
    )

    for s in states:
        print(f"s.messages():{s.messages()}")


if __name__ == "__main__":
    model_path = "/data/llama3/Meta-Llama-3-8B-Instruct-hf"
    runtime = sgl.Runtime(model_path=model_path)
    sgl.set_default_backend(runtime)

    # Run a single request
    # print("\n========== single ==========\n")
    # single()

    # # Stream output
    # print("\n========== stream ==========\n")
    # stream()

    # Run a batch of requests
    print("\n========== batch ==========\n")
    batch()

    runtime.shutdown()
