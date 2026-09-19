from logic_engine import KnowledgeBase


def test_safe_to_engage():
    kb = KnowledgeBase()

    # Rule:
    # TargetVisible AND HasDust -> SafeToEngage
    kb.tell_rule(
        ['TargetVisible', 'HasDust'],
        'SafeToEngage'
    )

    # Add required facts
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')

    # Run forward chaining
    kb.forward_chain()

    # SafeToEngage should be inferred
    assert 'SafeToEngage' in kb.facts

    # Retreat should NOT be inferred
    assert 'Retreat' not in kb.facts


def test_retreat():
    kb = KnowledgeBase()

    # Rule 1
    kb.tell_rule(
        ['TargetVisible', 'HasDust'],
        'SafeToEngage'
    )

    # Rule 2
    kb.tell_rule(
        ['SafeToEngage', 'BloodseekerMissing'],
        'Retreat'
    )

    # Add all required facts
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.tell_fact('BloodseekerMissing')

    # Run forward chaining
    kb.forward_chain()

    # SafeToEngage should be inferred first
    assert 'SafeToEngage' in kb.facts

    # Then Retreat should be inferred
    assert 'Retreat' in kb.facts


if __name__ == "__main__":

    test_safe_to_engage()
    test_retreat()

    print("✅ All Logic Engine Test Cases Passed!")