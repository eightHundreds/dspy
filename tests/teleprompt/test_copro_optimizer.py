import dspy
from dspy import Example, InputField, OutputField
from dspy.teleprompt.signature_opt import COPRO
from dspy.utils.dummies import DummyLM
from dspy.signatures.signature import ensure_signature, Signature


# Define a simple metric function for testing
def simple_metric(example, prediction):
    # Simplified metric for testing: true if prediction matches expected output
    return example.output == prediction.output


# Example training and validation sets
trainset = [
    Example(input="Question: What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="Question: What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs(
        "input"
    ),
]


def test_signature_optimizer_initialization():
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert optimizer.breadth == 2, "Breadth not correctly initialized"
    assert optimizer.depth == 1, "Depth not correctly initialized"
    assert optimizer.init_temperature == 1.4, "Initial temperature not correctly initialized"


class SimpleSignature(Signature):
    input: str = InputField()
    output: str = OutputField()


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        # COPRO doesn't work with dspy.Predict
        if isinstance(signature, str):
            signature = SimpleSignature
        self.predictor = dspy.ChainOfThought(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def test_signature_optimizer_optimization_process():
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    dspy.settings.configure(
        lm=DummyLM(
            [
                {
                    "proposed_instruction": "Optimized instruction 1",
                    "proposed_prefix_for_output_field": "Optimized instruction 2",
                },
            ]
        )
    )

    student = SimpleModule("input -> output")

    # Assuming the compile method of COPRO requires a student module, a development set, and evaluation kwargs
    optimized_student = optimizer.compile(
        student, trainset=trainset, eval_kwargs={"num_threads": 1, "display_progress": False}
    )

    # Check that the optimized student has been modified from the original
    # This check can be more specific based on how the optimization modifies the student
    assert optimized_student is not student, "Optimization did not modify the student"

    # Further tests can be added to verify the specifics of the optimization process,
    # such as checking the instructions of the optimized student's predictors.


def test_signature_optimizer_statistics_tracking():
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    optimizer.track_stats = True  # Enable statistics tracking

    dspy.settings.configure(
        lm=DummyLM(
            [
                {
                    "proposed_instruction": "Optimized instruction 1",
                    "proposed_prefix_for_output_field": "Optimized instruction 2",
                },
            ]
        )
    )
    student = SimpleModule("input -> output")
    optimized_student = optimizer.compile(
        student, trainset=trainset, eval_kwargs={"num_threads": 1, "display_progress": False}
    )

    # Verify that statistics have been tracked and attached to the optimized student
    assert hasattr(optimized_student, "total_calls"), "Total calls statistic not tracked"
    assert hasattr(optimized_student, "results_best"), "Best results statistics not tracked"


# Assuming the setup_signature_optimizer fixture and simple_metric function are defined as before


def test_optimization_and_output_verification():
    lm = DummyLM(
        [
            {"proposed_instruction": "Optimized Prompt", "proposed_prefix_for_output_field": "Optimized Prefix"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
        ]
    )
    dspy.settings.configure(lm=lm)
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)

    student = SimpleModule("input -> output")

    # Compile the student with the optimizer
    optimized_student = optimizer.compile(
        student, trainset=trainset, eval_kwargs={"num_threads": 1, "display_progress": False}
    )

    # Simulate calling the optimized student with a new input
    test_input = "What is the capital of France?"
    prediction = optimized_student(input=test_input)

    print(lm.get_convo(-1))

    assert prediction.output == "Paris"


def test_statistics_tracking_during_optimization():
    dspy.settings.configure(
        lm=DummyLM(
            [
                {"proposed_instruction": "Optimized Prompt", "proposed_prefix_for_output_field": "Optimized Prefix"},
            ]
        )
    )

    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    optimizer.track_stats = True  # Enable statistics tracking

    student = SimpleModule("input -> output")
    optimized_student = optimizer.compile(
        student, trainset=trainset, eval_kwargs={"num_threads": 1, "display_progress": False}
    )

    # Verify that statistics have been tracked
    assert hasattr(optimized_student, "total_calls"), "Optimizer did not track total metric calls"
    assert optimized_student.total_calls > 0, "Optimizer reported no metric calls"

    # Check if the results_best and results_latest contain valid statistics
    assert "results_best" in optimized_student.__dict__, "Optimizer did not track the best results"
    assert "results_latest" in optimized_student.__dict__, "Optimizer did not track the latest results"
    assert len(optimized_student.results_best) > 0, "Optimizer did not properly populate the best results statistics"
    assert (
        len(optimized_student.results_latest) > 0
    ), "Optimizer did not properly populate the latest results statistics"

    # Additional detailed checks can be added here to verify the contents of the tracked statistics


class TestSignature(dspy.Signature):
    """A simple test signature."""

    input = dspy.InputField()
    output = dspy.OutputField()


def test_copro_with_complex_evaluation_output():
    """Test the case when both return_outputs=True and return_all_scores=True"""
    dspy.settings.configure(
        lm=DummyLM(
            {
                # For predictions after optimization
                "What is the color of the sky?": {
                    "reasoning": "让我们一步一步思考。\n1. 这是一个简单的问题\n2. 天空的颜色是蓝色的\n3. 所以答案是蓝色",
                    "output": "blue",
                },
                "Question: What is the color of the sky?": {
                    "reasoning": "让我们一步一步思考。\n1. 这是一个简单的问题\n2. 天空的颜色是蓝色的\n3. 所以答案是蓝色",
                    "output": "blue",
                },
                "Question: What does the fox say?": {
                    "reasoning": "让我们听听狐狸的声音。\n1. 这是一个有趣的问题\n2. 根据流行歌曲\n3. 狐狸说 Ring-ding-ding-ding-dingeringeding!",
                    "output": "Ring-ding-ding-ding-dingeringeding!",
                },
                # For COPRO optimization process
                "basic_instruction": {
                    "proposed_instruction": "Given the input question, think step by step to provide a clear and accurate answer.",
                    "proposed_prefix_for_output_field": "Let's solve this step by step:",
                },
                # For COPRO's GenerateInstructionGivenAttempts
                "attempted_instructions": {
                    "proposed_instruction": "Given the input question, analyze it carefully and provide a detailed answer.",
                    "proposed_prefix_for_output_field": "Let's analyze this:",
                },
            }
        )
    )

    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    student = SimpleModule("input -> output")  # Using string format signature

    # Compile student model with complex evaluation output enabled
    eval_kwargs = {"num_threads": 1, "display_progress": False, "return_outputs": True, "return_all_scores": True}

    # This call would fail before the fix
    optimized_student = optimizer.compile(student, trainset=trainset, eval_kwargs=eval_kwargs)

    # Verify optimization completed successfully
    assert hasattr(optimized_student, "total_calls"), "Optimization should complete and record total calls"

    # Make prediction with optimized model
    test_input = "What is the color of the sky?"
    prediction = optimized_student(input=test_input)
    assert isinstance(optimized_student.candidate_programs[0]["score"][0], (int, float))
    # Verify optimized model works properly
    assert hasattr(prediction, "output"), "Optimized model should generate predictions normally"
    assert prediction.output == "blue", "Prediction should be 'blue'"
    assert hasattr(prediction, "reasoning"), "Prediction should include reasoning process"
