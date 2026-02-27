#!/usr/bin/env python3
"""
Test disease/pest detection exactly as frontend does
"""
import requests
from PIL import Image
import io

BASE_URL = "http://127.0.0.1:5000"

# Login
login_resp = requests.post(f"{BASE_URL}/login", json={
    "email": "testuser@test.com",
    "password": "test123"
})

if login_resp.status_code != 200:
    print(f"Login failed: {login_resp.status_code}")
    print(login_resp.text)
    exit(1)

token = login_resp.json()["access_token"]
print(f"Token: {token[:20]}...")

# Test 1: Create test image and upload for disease detection
print("\n=== TEST 1: Disease Detection ===")
img = Image.new('RGB', (224, 224), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

headers = {"Authorization": f"Bearer {token}"}
files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}

try:
    resp = requests.post(
        f"{BASE_URL}/predict-disease",
        headers=headers,
        files=files,
        timeout=30
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
except Exception as e:
    print(f"Exception: {e}")
    print(f"Response text: {resp.text if 'resp' in locals() else 'N/A'}")

# Test 2: Create test image and upload for pest detection
print("\n=== TEST 2: Pest Detection ===")
img = Image.new('RGB', (224, 224), color='green')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

headers = {"Authorization": f"Bearer {token}"}
files = {'image': ('pest_test.jpg', img_bytes, 'image/jpeg')}

try:
    resp = requests.post(
        f"{BASE_URL}/predict-pest",
        headers=headers,
        files=files,
        timeout=30
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
except Exception as e:
    print(f"Exception: {e}")
    print(f"Response text: {resp.text if 'resp' in locals() else 'N/A'}")

print("\n=== TESTS COMPLETED ===")
