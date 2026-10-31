from gotrain import diary


def test_old_session_does_not_invent_rejection_or_remaining_budget():
    rows = [{"event": "tool_result", "tool": "run_experiment", "result": {
        "status": "completed", "selection_metric": "mean_iou", "selection_score": 0.84,
        "best_epoch": 15, "elapsed_seconds": 60,
    }}]
    messages, pending = diary.entries(rows)
    assert pending is None
    assert "completed (verdict not recorded)" in messages[0]["title"]
    assert "rejected" not in messages[0]["title"]
    assert messages[0]["tone"] == "info"
    assert "budget left" not in messages[0]["detail"]
