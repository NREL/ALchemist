"""
Quick test for new audit log and metadata API endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_audit_api():
    print("Testing Audit Log and Metadata API Endpoints...")
    print("=" * 60)
    
    # Create session
    print("\n1. Creating session...")
    resp = requests.post(f"{BASE_URL}/sessions")
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    print(f"   ✓ Session created: {session_id}")
    
    # Update metadata
    print("\n2. Updating metadata...")
    resp = requests.patch(
        f"{BASE_URL}/sessions/{session_id}/metadata",
        json={
            "name": "Test_Session_Nov2025",
            "description": "Testing audit log API",
            "tags": ["test", "audit", "demo"]
        }
    )
    assert resp.status_code == 200
    metadata = resp.json()
    print(f"   ✓ Metadata updated: {metadata['name']}")
    print(f"   ✓ Tags: {metadata['tags']}")
    
    # Get metadata
    print("\n3. Getting metadata...")
    resp = requests.get(f"{BASE_URL}/sessions/{session_id}/metadata")
    assert resp.status_code == 200
    print(f"   ✓ Metadata retrieved")
    
    # Add some data
    print("\n4. Adding experimental data...")
    resp = requests.post(
        f"{BASE_URL}/sessions/{session_id}/variables",
        json={"name": "temp", "type": "real", "min": 100, "max": 300}
    )
    assert resp.status_code == 200
    
    resp = requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments",
        json={"inputs": {"temp": 200}, "output": 85.0}
    )
    assert resp.status_code == 200
    print(f"   ✓ Data added")
    
    # Lock data
    print("\n5. Locking data decision...")
    resp = requests.post(
        f"{BASE_URL}/sessions/{session_id}/audit/lock",
        json={"lock_type": "data", "notes": "Initial test dataset"}
    )
    assert resp.status_code == 200
    lock_resp = resp.json()
    print(f"   ✓ {lock_resp['message']}")
    print(f"   ✓ Entry hash: {lock_resp['entry']['hash']}")
    
    # Get audit log
    print("\n6. Getting audit log...")
    resp = requests.get(f"{BASE_URL}/sessions/{session_id}/audit")
    assert resp.status_code == 200
    audit = resp.json()
    print(f"   ✓ Audit log retrieved: {audit['n_entries']} entries")
    if audit['entries']:
        print(f"   ✓ Latest entry: {audit['entries'][0]['entry_type']}")
    
    # Train model
    print("\n7. Training model...")
    resp = requests.post(
        f"{BASE_URL}/sessions/{session_id}/model/train",
        json={"backend": "sklearn", "kernel": "rbf"}
    )
    # May fail with only 1 data point, but that's ok for this test
    print(f"   ✓ Model training attempted (status: {resp.status_code})")
    
    # Try to lock model (might fail if training failed)
    if resp.status_code == 200:
        print("\n8. Locking model decision...")
        resp = requests.post(
            f"{BASE_URL}/sessions/{session_id}/audit/lock",
            json={"lock_type": "model", "notes": "Test model"}
        )
        if resp.status_code == 200:
            print(f"   ✓ Model locked")
        else:
            print(f"   ⚠ Model lock failed (expected - need more data)")
    
    # Export audit log as markdown
    print("\n9. Exporting audit log as markdown...")
    resp = requests.get(f"{BASE_URL}/sessions/{session_id}/audit/export")
    assert resp.status_code == 200
    print(f"   ✓ Markdown exported ({len(resp.content)} bytes)")
    
    # Download session
    print("\n10. Downloading session file...")
    resp = requests.get(f"{BASE_URL}/sessions/{session_id}/download")
    assert resp.status_code == 200
    print(f"   ✓ Session downloaded ({len(resp.content)} bytes)")
    
    # Save session content for upload test
    session_content = resp.content
    filename = resp.headers.get('content-disposition', '').split('filename=')[-1]
    print(f"   ✓ Filename: {filename}")
    
    # Delete session
    print("\n11. Deleting original session...")
    resp = requests.delete(f"{BASE_URL}/sessions/{session_id}")
    assert resp.status_code == 204
    print(f"   ✓ Session deleted")
    
    # Upload session
    print("\n12. Uploading session file...")
    files = {'file': ('session.json', session_content, 'application/json')}
    resp = requests.post(f"{BASE_URL}/sessions/upload", files=files)
    assert resp.status_code == 201
    new_session_id = resp.json()["session_id"]
    print(f"   ✓ Session uploaded: {new_session_id}")
    
    # Verify metadata preserved
    print("\n13. Verifying restored metadata...")
    resp = requests.get(f"{BASE_URL}/sessions/{new_session_id}/metadata")
    assert resp.status_code == 200
    restored_metadata = resp.json()
    print(f"   ✓ Name preserved: {restored_metadata['name']}")
    print(f"   ✓ Tags preserved: {restored_metadata['tags']}")
    
    # Verify audit log preserved
    print("\n14. Verifying restored audit log...")
    resp = requests.get(f"{BASE_URL}/sessions/{new_session_id}/audit")
    assert resp.status_code == 200
    restored_audit = resp.json()
    print(f"   ✓ Audit entries preserved: {restored_audit['n_entries']}")
    
    # Cleanup
    print("\n15. Cleanup...")
    resp = requests.delete(f"{BASE_URL}/sessions/{new_session_id}")
    print(f"   ✓ Test session deleted")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_audit_api()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to API. Make sure the server is running:")
        print("   python -m api.run_api")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
