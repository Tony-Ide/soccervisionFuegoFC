import cv2
import numpy as np
import time
import requests

# Stream URL
STITCHED_STREAM_URL = "http://localhost:5000/api/stitched/stream"

def test_stream_with_reconnection():
    """Test the stitched stream with automatic reconnection"""
    print("🔗 Testing stitched stream with reconnection...")
    
    max_retries = 5
    reconnect_delay = 2
    
    for attempt in range(max_retries):
        print(f"📡 Attempt {attempt + 1}/{max_retries}: Connecting to stream...")
        
        try:
            # Try to connect to the stream
            cap = cv2.VideoCapture(STITCHED_STREAM_URL)
            if not cap.isOpened():
                print(f"❌ Attempt {attempt + 1}: Failed to open stream")
                time.sleep(reconnect_delay)
                continue
            
            print(f"✅ Attempt {attempt + 1}: Connected to stream!")
            
            # Test reading frames
            frame_count = 0
            start_time = time.time()
            
            while frame_count < 100:  # Test 100 frames
                ret, frame = cap.read()
                if not ret:
                    print(f"❌ Stream ended after {frame_count} frames")
                    break
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"📷 Got frame {frame_count}")
                
                # Display frame
                cv2.imshow('Stitched Stream Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"📊 Test completed: {frame_count} frames in {elapsed_time:.2f}s ({fps:.1f} FPS)")
            
            cap.release()
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            time.sleep(reconnect_delay)
    
    print("❌ All connection attempts failed")
    return False

def test_stream_status():
    """Test the stream status API"""
    print("📊 Testing stream status...")
    
    try:
        response = requests.get('http://localhost:5000/api/stitch/status', timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("✅ Stream status:")
            print(f"   - Stitching enabled: {status.get('enabled', 'Unknown')}")
            print(f"   - Homography locked: {status.get('locked', 'Unknown')}")
            print(f"   - Cameras available: {status.get('cameras', 'Unknown')}")
            return True
        else:
            print(f"❌ Status API returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Status API failed: {e}")
    
    return False

def main():
    """Main test function"""
    print("🧪 Starting stitched stream tests...")
    print("=" * 50)
    
    # Test 1: Stream status
    print("\n1️⃣ Testing stream status...")
    status_ok = test_stream_status()
    
    # Test 2: Stream with reconnection
    print("\n2️⃣ Testing stream with reconnection...")
    stream_ok = test_stream_with_reconnection()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Status API: {'✅ PASS' if status_ok else '❌ FAIL'}")
    print(f"   Stream: {'✅ PASS' if stream_ok else '❌ FAIL'}")
    
    if stream_ok:
        print("\n🎉 Stream is working! You can use it in main.py")
    else:
        print("\n⚠️ Stream has issues. Check your stitching API server.")

if __name__ == "__main__":
    main()