#!/bin/bash

echo "🔧 Fixing TA-Lib linking issue..."

# Step 1: Create a symbolic link for the library name expected by pip
echo "📎 Creating symbolic link for library..."
sudo ln -sf /usr/local/lib/libta_lib.a /usr/local/lib/libta-lib.a
sudo ln -sf /usr/local/lib/libta_lib.so /usr/local/lib/libta-lib.so
sudo ln -sf /usr/local/lib/libta_lib.so.0 /usr/local/lib/libta-lib.so.0
sudo ln -sf /usr/local/lib/libta_lib.so.0.0.0 /usr/local/lib/libta-lib.so.0.0.0

# Step 2: Update library cache
echo "🔄 Updating library cache..."
sudo ldconfig

# Step 3: Verify the symbolic links
echo "✅ Verifying symbolic links..."
ls -la /usr/local/lib/libta*

# Step 4: Try installing Python TA-Lib again
echo "🐍 Installing Python TA-Lib wrapper (attempt 2)..."
pip install --no-cache-dir TA-Lib

# Step 5: Test installation
echo "🧪 Testing TA-Lib installation..."
python3 -c "
import talib
import numpy as np
print('✅ TA-Lib successfully imported')
print(f'TA-Lib version: {talib.__version__}')

# Test a simple function
data = np.random.random(100)
sma = talib.SMA(data, timeperiod=20)
print('✅ TA-Lib functions working correctly')
print('🎉 TA-Lib installation successful!')
"

echo "🏁 TA-Lib fix complete!"
