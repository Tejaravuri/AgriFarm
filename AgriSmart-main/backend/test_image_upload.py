#!/usr/bin/env python3
"""
Test disease and pest detection with actual file uploads
"""
import requests
import json
from PIL import Image
import io
import time

# Wait for Flask server to start
time.sleep(2)

BASE_URL = "http://127.0.0.1:5000"

# Step 1: Signup and login
print("Step 1: Register user")
signup_resp = requests.post(f"{BASE_URL}/signup", json={
    "name": "Test User",
    "email": "testuser@test.com",
    "password": "test123"
})
print(f"Signup status: {signup_resp.status_code}")

print("\nStep 2: Login")
login_resp = requests.post(f"{BASE_URL}/login", json={
    "email": "testuser@test.com",
    "password": "test123"
})
print(f"Login status: {login_resp.status_code}")

if login_resp.status_code != 200:
    print(f"Login failed: {login_resp.text}")
    exit(1)

token = login_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Step 3: Test disease detection
print("\n" + "=" * 50)
print("Step 3: Test DISEASE DETECTION")
print("=" * 50)

# Create a test image
img = Image.new('RGB', (200, 200), color='red')
img_io = io.BytesIO()
img.save(img_io, format='JPEG')
img_io.seek(0)

files = {'image': ('test_disease.jpg', img_io, 'image/jpeg')}
disease_resp = requests.post(f"{BASE_URL}/predict-disease", headers=headers, files=files)
print(f"Status: {disease_resp.status_code}")
print(f"Response: {disease_resp.text}")

if disease_resp.status_code != 200:
    print("ERROR in disease detection!")
    try:
        print(f"Error message: {disease_resp.json()}")
    except:
        print(f"Raw response: {disease_resp.text}")

# Step 4: Test pest detection
print("\n" + "=" * 50)
print("Step 4: Test PEST DETECTION")
print("=" * 50)

# Create a test image
img = Image.new('RGB', (200, 200), color='green')
img_io = io.BytesIO()
img.save(img_io, format='JPEG')
img_io.seek(0)

files = {'image': ('test_pest_aphids.jpg', img_io, 'image/jpeg')}
pest_resp = requests.post(f"{BASE_URL}/predict-pest", headers=headers, files=files)
print(f"Status: {pest_resp.status_code}")
print(f"Response: {pest_resp.text}")

if pest_resp.status_code != 200:
    print("ERROR in pest detection!")
    try:
        print(f"Error message: {pest_resp.json()}")
    except:
        print(f"Raw response: {pest_resp.text}")

print("\n[TEST COMPLETED]")
