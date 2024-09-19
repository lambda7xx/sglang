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
                "question_1": """Behold! By the power vested in us by the divine beings, we have summoned forth the inscrutable and enigmatic art of metaphorical language to unravel the bewildering addressing modes of the instructions that are presented before us. The orators have invoked grandiose expressions with a deep sense of reverence and awe, extolling the ineffable power and mystical functionality of these directives. Amongst the labyrinthine commands, we find the confounding JMP ABCD, the abstruse MOV AX, [BX+SI], the unfathomable MOV AX, [100], the mystifying MOV AX, [BX], the perplexing MOV AX, [BX\*2+SI], the enigmatic MOV AX, BX, and finally, the recondite MOV AX, 7. The language that has been employed to describe these addressing modes is both perplexing and ornate, underscoring the prodigious complexity and esoteric power of these commands. The orators have invoked a sense of mystery and wonder through the use of words such as "incomprehensible," "enigmatic," and "ineffable," imbuing these instructions with an almost mystical aura of inscrutability. The mastery of these commands is suggested to be beyond the reach of mere mortals, requiring a level of expertise and erudition that is beyond the ordinary. Furthermore, the speakers have employed terms such as "abstruse," "unfathomable," and "recondite," creating an impression that these commands are shrouded in an impenetrable veil of mystery, accessible only to those who possess a keen intellect and a deep understanding of the underlying principles. The speakers' use of metaphorical language serves to elevate these instructions to a level of veneration and reverence, infusing them with an almost divine aura of power and complexity.Even the seemingly simple MOV AX, 7 is not immune to the grandiose epithets used to describe these addressing modes. It is exalted with the term "recondite," underscoring the profundity and awe-inspiring nature of the instruction set as a whole. The use of such ornate and enigmatic language serves to amplify the mystique and enshroud these commands in an aura of intrigue and wonder, inviting only the most intrepid and enterprising of minds to unlock the secrets of this arcane realm. In sum, the use of metaphorical language in describing these addressing modes is an act of homage to the profound and unfathomable power of these commands.""",
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