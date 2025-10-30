"""
Simple test runner for ALchemist Core event system.
This verifies the EventEmitter functionality without requiring pytest.
"""

import sys
import os

# Add alchemist_core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_import():
    """Test that we can import EventEmitter"""
    print("Testing EventEmitter import...")
    
    try:
        from alchemist_core import EventEmitter
        print("✓ Successfully imported EventEmitter from alchemist_core")
    except ImportError as e:
        print(f"✗ Failed to import EventEmitter: {e}")
        return False
    
    try:
        from alchemist_core.events import EventEmitter as EE
        print("✓ Successfully imported from alchemist_core.events")
    except ImportError as e:
        print(f"✗ Failed to import from alchemist_core.events: {e}")
        return False
    
    return True


def test_basic_event_emit():
    """Test basic event emission and listening"""
    print("\nTesting basic event emission...")
    
    from alchemist_core import EventEmitter
    
    emitter = EventEmitter()
    
    # Track if callback was called
    called_data = []
    
    def callback(data):
        called_data.append(data)
    
    # Subscribe
    emitter.on("test_event", callback)
    
    # Emit event
    emitter.emit("test_event", {"value": 42})
    
    # Verify callback was called
    assert len(called_data) == 1, "Callback should have been called once"
    assert called_data[0]["value"] == 42, "Callback should receive correct data"
    
    print("✓ Basic event emission works")
    return True


def test_multiple_listeners():
    """Test multiple listeners for same event"""
    print("\nTesting multiple listeners...")
    
    from alchemist_core import EventEmitter
    
    emitter = EventEmitter()
    
    call_count = [0, 0]  # Track calls for two listeners
    
    def listener1(data):
        call_count[0] += 1
    
    def listener2(data):
        call_count[1] += 1
    
    emitter.on("event", listener1)
    emitter.on("event", listener2)
    
    # Emit once
    emitter.emit("event", {})
    
    assert call_count[0] == 1, "Listener 1 should be called"
    assert call_count[1] == 1, "Listener 2 should be called"
    
    print("✓ Multiple listeners work")
    return True


def test_unsubscribe():
    """Test unsubscribing from events"""
    print("\nTesting unsubscribe...")
    
    from alchemist_core import EventEmitter
    
    emitter = EventEmitter()
    
    call_count = [0]
    
    def callback(data):
        call_count[0] += 1
    
    # Subscribe
    emitter.on("event", callback)
    emitter.emit("event", {})
    assert call_count[0] == 1
    
    # Unsubscribe
    emitter.off("event", callback)
    emitter.emit("event", {})
    
    # Should still be 1 (not called again)
    assert call_count[0] == 1, "Callback should not be called after unsubscribe"
    
    print("✓ Unsubscribe works")
    return True


def test_once():
    """Test once() method for one-time listeners"""
    print("\nTesting once() method...")
    
    from alchemist_core import EventEmitter
    
    emitter = EventEmitter()
    
    call_count = [0]
    
    def callback(data):
        call_count[0] += 1
    
    # Subscribe with once
    emitter.once("event", callback)
    
    # Emit multiple times
    emitter.emit("event", {})
    emitter.emit("event", {})
    emitter.emit("event", {})
    
    # Should only be called once
    assert call_count[0] == 1, "once() callback should only fire once"
    
    print("✓ once() method works")
    return True


def test_error_handling():
    """Test that errors in listeners don't break the emitter"""
    print("\nTesting error handling...")
    
    from alchemist_core import EventEmitter
    
    emitter = EventEmitter()
    
    call_count = [0]
    
    def bad_callback(data):
        raise ValueError("Intentional error")
    
    def good_callback(data):
        call_count[0] += 1
    
    # Subscribe both (bad one first)
    emitter.on("event", bad_callback)
    emitter.on("event", good_callback)
    
    # Emit event - should not raise, good callback should still run
    emitter.emit("event", {})
    
    assert call_count[0] == 1, "Good callback should run despite error in other callback"
    
    print("✓ Error handling works (errors in listeners don't break emitter)")
    return True


def test_listener_count():
    """Test listener_count() method"""
    print("\nTesting listener_count()...")
    
    from alchemist_core import EventEmitter
    
    emitter = EventEmitter()
    
    def callback1(data):
        pass
    
    def callback2(data):
        pass
    
    assert emitter.listener_count("event") == 0, "Should start with 0 listeners"
    
    emitter.on("event", callback1)
    assert emitter.listener_count("event") == 1, "Should have 1 listener"
    
    emitter.on("event", callback2)
    assert emitter.listener_count("event") == 2, "Should have 2 listeners"
    
    emitter.off("event", callback1)
    assert emitter.listener_count("event") == 1, "Should have 1 listener after removing one"
    
    print("✓ listener_count() works")
    return True


def test_remove_all_listeners():
    """Test remove_all_listeners() method"""
    print("\nTesting remove_all_listeners()...")
    
    from alchemist_core import EventEmitter
    
    emitter = EventEmitter()
    
    def callback(data):
        pass
    
    emitter.on("event1", callback)
    emitter.on("event2", callback)
    
    # Remove all listeners for event1
    emitter.remove_all_listeners("event1")
    assert emitter.listener_count("event1") == 0
    assert emitter.listener_count("event2") == 1
    
    # Remove all listeners for all events
    emitter.remove_all_listeners()
    assert emitter.listener_count("event1") == 0
    assert emitter.listener_count("event2") == 0
    
    print("✓ remove_all_listeners() works")
    return True


def test_event_names():
    """Test event_names() method"""
    print("\nTesting event_names()...")
    
    from alchemist_core import EventEmitter
    
    emitter = EventEmitter()
    
    def callback(data):
        pass
    
    assert emitter.event_names() == [], "Should start with no events"
    
    emitter.on("event1", callback)
    emitter.on("event2", callback)
    
    names = emitter.event_names()
    assert "event1" in names, "Should include event1"
    assert "event2" in names, "Should include event2"
    assert len(names) == 2, "Should have 2 event names"
    
    print("✓ event_names() works")
    return True


def test_progress_event_pattern():
    """Test realistic progress event pattern"""
    print("\nTesting realistic progress event pattern...")
    
    from alchemist_core import EventEmitter
    
    # Simulate a model training process
    emitter = EventEmitter()
    
    progress_updates = []
    
    def track_progress(data):
        progress_updates.append(data)
    
    emitter.on("progress", track_progress)
    
    # Simulate training loop
    total_iterations = 5
    for i in range(total_iterations):
        emitter.emit("progress", {
            "current": i + 1,
            "total": total_iterations,
            "message": f"Training iteration {i+1}"
        })
    
    assert len(progress_updates) == 5, "Should have 5 progress updates"
    assert progress_updates[0]["current"] == 1
    assert progress_updates[-1]["current"] == 5
    
    print("✓ Realistic progress event pattern works")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("ALchemist Core - Event System Tests")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        ("Import Test", test_import),
        ("Basic Event Emission", test_basic_event_emit),
        ("Multiple Listeners", test_multiple_listeners),
        ("Unsubscribe", test_unsubscribe),
        ("Once Method", test_once),
        ("Error Handling", test_error_handling),
        ("Listener Count", test_listener_count),
        ("Remove All Listeners", test_remove_all_listeners),
        ("Event Names", test_event_names),
        ("Progress Event Pattern", test_progress_event_pattern),
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                print(f"\n✗ {test_name} FAILED")
                all_passed = False
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
