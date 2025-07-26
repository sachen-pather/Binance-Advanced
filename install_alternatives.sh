#!/bin/bash

echo "📦 Installing alternative technical analysis libraries..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Install alternative libraries
echo "🐍 Installing pandas-ta..."
pip install pandas-ta

echo "🐍 Installing ta..."
pip install ta

echo "🐍 Installing finta..."
pip install finta

# Test installations
echo "🧪 Testing alternative libraries..."
python3 -c "
print('Testing alternative technical analysis libraries...')

success_count = 0
total_libraries = 3

# Test pandas-ta
try:
    import pandas_ta as pta
    import pandas as pd
    import numpy as np
    
    # Create sample data
    data = pd.Series(np.random.random(100) * 100 + 50)
    sma = pta.sma(data, length=20)
    
    print('✅ pandas-ta: Working correctly')
    success_count += 1
except Exception as e:
    print(f'❌ pandas-ta: {e}')

# Test ta library
try:
    import ta
    import pandas as pd
    import numpy as np
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'high': np.random.random(100) * 100 + 50,
        'low': np.random.random(100) * 100 + 40,
        'close': np.random.random(100) * 100 + 45,
        'volume': np.random.random(100) * 1000000
    })
    
    rsi = ta.momentum.RSIIndicator(df['close']).rsi()
    print('✅ ta library: Working correctly')
    success_count += 1
except Exception as e:
    print(f'❌ ta library: {e}')

# Test finta
try:
    import finta
    import pandas as pd
    import numpy as np
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'high': np.random.random(100) * 100 + 50,
        'low': np.random.random(100) * 100 + 40,
        'close': np.random.random(100) * 100 + 45,
        'volume': np.random.random(100) * 1000000
    })
    
    sma = finta.TA.SMA(df, 20)
    print('✅ finta: Working correctly')
    success_count += 1
except Exception as e:
    print(f'❌ finta: {e}')

print(f'\\n🎉 {success_count}/{total_libraries} alternative libraries working!')

if success_count >= 1:
    print('✅ You have at least one working alternative to TA-Lib!')
    print('📄 Your enhanced indicators module will work properly.')
else:
    print('⚠️  No alternative libraries working. Manual calculations will be used.')
"

echo "🏁 Alternative library installation complete!"
