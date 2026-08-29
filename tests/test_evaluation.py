from sciagent.evaluation.metrics import compare_tool_sequence, evaluate_agent_result


def test_exact_tool_sequence():
    result = compare_tool_sequence(
        ["detect_zscore_anomalies", "welch_t_test", "pearson_correlation"],
        ["detect_zscore_anomalies", "welch_t_test", "pearson_correlation"],
    )
    assert result["exact_match"] is True
    assert result["sequence_accuracy"] == 1.0


def test_partial_tool_sequence():
    result = evaluate_agent_result(
        observed_tools=["detect_zscore_anomalies"],
        expected_tools=["detect_zscore_anomalies", "welch_t_test", "pearson_correlation"],
        conclusion="An anomalous period exists.",
    )
    assert result["tool_sequence"]["matching_positions"] == 1
    assert result["tool_sequence"]["sequence_accuracy"] < 1.0
    assert result["has_conclusion"] is True
