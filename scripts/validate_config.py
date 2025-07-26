#!/usr/bin/env python3
"""
Comprehensive configuration validation script for Enhanced Trading Bot
Validates all components, dependencies, and configuration before launch.
Updated to support alternative technical analysis libraries.
"""

import os
import sys
import json
import importlib
import traceback
from pathlib import Path
from datetime import datetime

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def print_step(step_num, text):
    """Print step message"""
    print(f"\n{Colors.BOLD}{Colors.PURPLE}Step {step_num}: {text}{Colors.END}")

def check_python_version():
    """Check Python version compatibility"""
    print_step(1, "Checking Python Version")
    
    major, minor = sys.version_info[:2]
    python_version = f"{major}.{minor}"
    
    if major < 3 or (major == 3 and minor < 8):
        print_error(f"Python {python_version} detected. Python 3.8+ required")
        print_info("Please upgrade Python to version 3.8 or higher")
        return False
    
    print_success(f"Python {python_version} - Compatible")
    return True

def check_directory_structure():
    """Validate directory structure"""
    print_step(2, "Checking Directory Structure")
    
    required_dirs = [
        'src', 'config', 'data', 'logs', 'backups', 'reports', 'tests', 'scripts'
    ]
    
    optional_dirs = [
        'data/historical', 'data/models', 'data/cache', 'data/exports',
        'backups/strategy_states', 'backups/databases', 'backups/models',
        'reports/daily', 'reports/weekly', 'reports/monthly', 'reports/backtests'
    ]
    
    all_good = True
    
    # Check required directories
    for directory in required_dirs:
        if os.path.exists(directory):
            print_success(f"Directory exists: {directory}")
        else:
            print_error(f"Missing required directory: {directory}")
            all_good = False
    
    # Check optional directories (create if missing)
    for directory in optional_dirs:
        if os.path.exists(directory):
            print_success(f"Optional directory exists: {directory}")
        else:
            print_warning(f"Creating optional directory: {directory}")
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print_success(f"Created directory: {directory}")
            except Exception as e:
                print_error(f"Failed to create directory {directory}: {e}")
    
    return all_good

def check_core_dependencies():
    """Check core Python dependencies"""
    print_step(3, "Checking Core Dependencies")
    
    core_deps = {
        'pandas': 'Data manipulation library',
        'numpy': 'Numerical computing library',
        'requests': 'HTTP library',
        'asyncio': 'Asynchronous I/O library',
        'json': 'JSON processing library',
        'datetime': 'Date and time library',
        'logging': 'Logging library',
        'pathlib': 'Path handling library',
        'sqlite3': 'SQLite database library'
    }
    
    all_good = True
    
    for package, description in core_deps.items():
        try:
            importlib.import_module(package)
            print_success(f"{package} - {description}")
        except ImportError:
            print_error(f"Missing core dependency: {package} - {description}")
            all_good = False
    
    return all_good

def check_trading_dependencies():
    """Check trading-specific dependencies with enhanced TA library support"""
    print_step(4, "Checking Trading Dependencies")
    
    trading_deps = {
        'binance': 'Binance API client',
        'scikit-learn': 'Machine learning library (sklearn)',
        'matplotlib': 'Plotting library',
        'seaborn': 'Statistical plotting library',
        'dotenv': 'Environment variable loader (python-dotenv)'
    }
    
    optional_deps = {
        'xgboost': 'XGBoost machine learning library',
        'lightgbm': 'LightGBM machine learning library',
        'optuna': 'Hyperparameter optimization library',
        'tensorflow': 'TensorFlow deep learning library'
    }
    
    # Technical Analysis Libraries (with priority order)
    ta_libraries = {
        'ta': 'Technical Analysis Library (Primary)',
        'pandas_ta': 'Pandas TA (Alternative)',
        'finta': 'FinTA (Alternative)',
        'talib': 'TA-Lib (Legacy)'
    }
    
    all_good = True
    
    # Check required trading dependencies
    for package, description in trading_deps.items():
        try:
            if package == 'binance':
                from binance.client import Client
                print_success(f"python-binance - {description}")
            elif package == 'scikit-learn':
                import sklearn
                print_success(f"scikit-learn - {description}")
            elif package == 'dotenv':
                from dotenv import load_dotenv
                print_success(f"python-dotenv - {description}")
            else:
                importlib.import_module(package)
                print_success(f"{package} - {description}")
        except ImportError as e:
            print_error(f"Missing trading dependency: {package} - {description}")
            print_info(f"Install with: pip install {package if package != 'scikit-learn' else 'scikit-learn'}")
            all_good = False
    
    # Check optional ML dependencies
    print_info("Checking optional advanced dependencies:")
    for package, description in optional_deps.items():
        try:
            importlib.import_module(package)
            print_success(f"{package} - {description} (OPTIONAL)")
        except ImportError:
            print_warning(f"Optional dependency missing: {package} - {description}")
            print_info(f"Install with: pip install {package}")
    
    # Check Technical Analysis Libraries
    print_info("Checking technical analysis libraries:")
    ta_libs_available = []
    ta_libs_working = []
    
    for package, description in ta_libraries.items():
        try:
            if package == 'ta':
                import ta
                # Test basic functionality
                import pandas as pd
                import numpy as np
                test_data = pd.DataFrame({
                    'high': np.random.random(50) * 100 + 50,
                    'low': np.random.random(50) * 100 + 40,
                    'close': np.random.random(50) * 100 + 45,
                    'volume': np.random.random(50) * 1000000
                })
                rsi = ta.momentum.RSIIndicator(test_data['close']).rsi()
                print_success(f"{package} - {description} ✨")
                ta_libs_available.append(package)
                ta_libs_working.append(package)
                
            elif package == 'pandas_ta':
                import pandas_ta as pta
                # Test basic functionality
                import pandas as pd
                import numpy as np
                test_data = pd.Series(np.random.random(50) * 100 + 50)
                sma = pta.sma(test_data, length=20)
                print_success(f"{package} - {description}")
                ta_libs_available.append(package)
                ta_libs_working.append(package)
                
            elif package == 'finta':
                import finta
                # Test with proper DataFrame structure
                import pandas as pd
                import numpy as np
                test_data = pd.DataFrame({
                    'open': np.random.random(50) * 100 + 50,
                    'high': np.random.random(50) * 100 + 50,
                    'low': np.random.random(50) * 100 + 40,
                    'close': np.random.random(50) * 100 + 45,
                    'volume': np.random.random(50) * 1000000
                })
                sma = finta.TA.SMA(test_data, 20)
                print_success(f"{package} - {description}")
                ta_libs_available.append(package)
                ta_libs_working.append(package)
                
            elif package == 'talib':
                import talib
                # Test if TA-Lib functions work properly
                import numpy as np
                test_data = np.random.random(50) * 100 + 50
                sma = talib.SMA(test_data, timeperiod=20)
                print_success(f"{package} - {description}")
                ta_libs_available.append(package)
                ta_libs_working.append(package)
                
        except ImportError:
            print_warning(f"TA library not installed: {package} - {description}")
            print_info(f"Install with: pip install {package}")
        except Exception as e:
            print_warning(f"TA library installed but not working: {package} - {e}")
            ta_libs_available.append(package)
            # Don't add to working list
    
    # Summary of TA libraries
    print_info(f"Technical Analysis Summary:")
    print(f"   📦 Libraries installed: {len(ta_libs_available)}")
    print(f"   ✅ Libraries working: {len(ta_libs_working)}")
    
    if ta_libs_working:
        print_success(f"Primary TA library: {ta_libs_working[0]}")
        if len(ta_libs_working) > 1:
            print_info(f"Backup libraries: {', '.join(ta_libs_working[1:])}")
    else:
        print_warning("No working TA libraries found - will use manual calculations")
        print_info("Recommend installing: pip install ta")
    
    return all_good

def check_environment_file():
    """Check .env file and environment variables"""
    print_step(5, "Checking Environment Configuration")
    
    if not os.path.exists('.env'):
        print_error("Missing .env file")
        print_info("Copy .env.example to .env and configure your settings")
        return False
    
    print_success("Found .env file")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print_success("Environment variables loaded")
    except Exception as e:
        print_error(f"Failed to load environment variables: {e}")
        return False
    
    # Check required environment variables
    required_vars = {
        'BINANCE_API_KEY': 'Binance API key',
        'BINANCE_SECRET_KEY': 'Binance secret key'
    }
    
    optional_vars = {
        'INITIAL_CAPITAL': 'Initial trading capital',
        'DEFAULT_PAPER_TRADE': 'Default paper trading mode',
        'LOG_LEVEL': 'Logging level',
        'MAX_DAILY_LOSS_PCT': 'Maximum daily loss percentage'
    }
    
    all_good = True
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value == f'your_{var.lower().replace("_", "_")}_here':
            print_error(f"Missing or not configured: {var} - {description}")
            all_good = False
        else:
            # Don't print actual API keys, just confirm they exist
            if 'key' in var.lower():
                print_success(f"{var} - Configured (length: {len(value)})")
            else:
                print_success(f"{var} - {value}")
    
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print_success(f"{var} - {value} (OPTIONAL)")
        else:
            print_warning(f"Optional variable not set: {var} - {description}")
    
    return all_good

def check_api_keys():
    """Validate API keys format"""
    print_step(6, "Validating API Keys")
    
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        print_error("API keys not found in environment variables")
        return False
    
    # Basic format validation
    if len(api_key) < 50:
        print_error("API key appears too short (should be ~64 characters)")
        return False
    
    if len(secret_key) < 50:
        print_error("Secret key appears too short (should be ~64 characters)")
        return False
    
    if api_key == 'your_api_key_here' or secret_key == 'your_secret_key_here':
        print_error("API keys still contain placeholder values")
        print_info("Please update .env file with your actual Binance API keys")
        return False
    
    print_success("API key format validation passed")
    return True

def test_api_connection():
    """Test Binance API connection"""
    print_step(7, "Testing API Connection")
    
    try:
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        client = Client(api_key, secret_key)
        
        # Test connection with server time (doesn't require authentication)
        print_info("Testing basic API connectivity...")
        server_time = client.get_server_time()
        print_success(f"API connection successful - Server time: {server_time}")
        
        # Test authenticated endpoint
        print_info("Testing authenticated API access...")
        try:
            account_info = client.get_account()
            print_success("Authenticated API access successful")
            
            # Check account permissions
            permissions = account_info.get('permissions', [])
            if 'SPOT' in permissions:
                print_success("Account has SPOT trading permissions")
            else:
                print_warning("Account may not have SPOT trading permissions")
                
        except BinanceAPIException as e:
            if 'Invalid API-key' in str(e):
                print_error("Invalid API key")
                return False
            elif 'Invalid signature' in str(e):
                print_error("Invalid API secret")
                return False
            elif 'IP not allowed' in str(e):
                print_error("IP address not whitelisted")
                print_info("Add your IP address to Binance API whitelist")
                return False
            else:
                print_warning(f"API authentication issue: {e}")
                print_info("Account info access failed, but basic connection works")
        
        return True
        
    except ImportError:
        print_error("python-binance library not installed")
        print_info("Install with: pip install python-binance")
        return False
    except Exception as e:
        print_error(f"API connection failed: {e}")
        print_info("Check your internet connection and API keys")
        return False

def check_enhanced_modules():
    """Check if enhanced trading modules can be imported with enhanced TA library testing"""
    print_step(8, "Checking Enhanced Trading Modules")
    
    # Add src to Python path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    enhanced_modules = {
        'enhanced_indicators': 'Enhanced Technical Indicators',
        'advanced_ml_engine': 'Advanced ML Engine',
        'advanced_market_analysis': 'Advanced Market Analysis',
        'advanced_risk_management': 'Advanced Risk Management',
        'enhanced_position_manager': 'Enhanced Position Manager',
        'comprehensive_analytics': 'Comprehensive Analytics',
        'enhanced_strategy_integration': 'Enhanced Strategy Integration',
        'advanced_backtesting_framework': 'Advanced Backtesting Framework',
        'enhanced_main_execution': 'Enhanced Main Execution'
    }
    
    all_good = True
    
    for module, description in enhanced_modules.items():
        try:
            imported_module = importlib.import_module(module)
            
            # Special test for enhanced_indicators module
            if module == 'enhanced_indicators':
                print_info("Testing enhanced indicators functionality...")
                try:
                    # Create test instance
                    indicators = imported_module.EnhancedTechnicalIndicators()
                    
                    # Create sample data for testing
                    import pandas as pd
                    import numpy as np
                    
                    np.random.seed(42)
                    test_data = pd.DataFrame({
                        'open': np.random.random(100) * 100 + 50,
                        'high': np.random.random(100) * 100 + 55,
                        'low': np.random.random(100) * 100 + 45,
                        'close': np.random.random(100) * 100 + 50,
                        'volume': np.random.randint(1000, 10000, 100)
                    })
                    
                    # Ensure OHLC relationships are correct
                    for i in range(len(test_data)):
                        row = test_data.iloc[i]
                        min_price = min(row['open'], row['close'])
                        max_price = max(row['open'], row['close'])
                        test_data.iloc[i, test_data.columns.get_loc('low')] = min(row['low'], min_price)
                        test_data.iloc[i, test_data.columns.get_loc('high')] = max(row['high'], max_price)
                    
                    # Test indicator calculation
                    result = indicators.calculate_all_indicators(test_data)
                    
                    if result is not None and len(result.columns) > 20:
                        indicator_count = len(result.columns) - 6  # Subtract original OHLCV + datetime
                        print_success(f"{module}.py - {description} ({indicator_count} indicators)")
                        
                        # Show which TA library is being used
                        if hasattr(imported_module, 'HAS_TA') and imported_module.HAS_TA:
                            print_info("   Using 'ta' library (optimal)")
                        elif hasattr(imported_module, 'HAS_PANDAS_TA') and imported_module.HAS_PANDAS_TA:
                            print_info("   Using 'pandas-ta' library")
                        elif hasattr(imported_module, 'HAS_FINTA') and imported_module.HAS_FINTA:
                            print_info("   Using 'finta' library")
                        elif hasattr(imported_module, 'HAS_TALIB') and imported_module.HAS_TALIB:
                            print_info("   Using 'talib' library")
                        else:
                            print_info("   Using manual calculations")
                    else:
                        print_warning(f"{module}.py - Limited functionality (few indicators calculated)")
                        
                except Exception as indicator_error:
                    print_warning(f"{module}.py - Import successful but calculation failed")
                    print_info(f"   Error: {indicator_error}")
            else:
                print_success(f"{module}.py - {description}")
                
        except ImportError as e:
            print_error(f"Cannot import {module}.py - {description}")
            print_info(f"Error: {e}")
            all_good = False
        except Exception as e:
            print_warning(f"Import issue with {module}.py - {description}")
            print_info(f"Error: {e}")
    
    return all_good

def check_file_permissions():
    """Check file and directory permissions"""
    print_step(9, "Checking File Permissions")
    
    # Check if key directories are writable
    writable_dirs = ['data', 'logs', 'backups', 'reports']
    all_good = True
    
    for directory in writable_dirs:
        if os.path.exists(directory):
            if os.access(directory, os.W_OK):
                print_success(f"Directory writable: {directory}")
            else:
                print_error(f"Directory not writable: {directory}")
                print_info(f"Fix with: chmod 755 {directory}")
                all_good = False
        else:
            print_warning(f"Directory does not exist: {directory}")
    
    # Check if .env file is readable
    if os.path.exists('.env'):
        if os.access('.env', os.R_OK):
            print_success(".env file readable")
        else:
            print_error(".env file not readable")
            all_good = False
    
    return all_good

def check_database_connection():
    """Check database connectivity"""
    print_step(10, "Checking Database Configuration")
    
    try:
        import sqlite3
        
        # Check if we can create/connect to SQLite database
        db_path = 'data/trading_bot.db'
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            conn.close()
            print_success("SQLite database connection successful")
            return True
        except Exception as e:
            print_error(f"Database connection failed: {e}")
            return False
            
    except ImportError:
        print_error("sqlite3 module not available")
        return False

def run_system_health_check():
    """Run basic system health check"""
    print_step(11, "System Health Check")
    
    try:
        import psutil
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent < 80:
            print_success(f"CPU usage: {cpu_percent}% (Good)")
        else:
            print_warning(f"CPU usage: {cpu_percent}% (High)")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent < 80:
            print_success(f"Memory usage: {memory.percent}% (Good)")
        else:
            print_warning(f"Memory usage: {memory.percent}% (High)")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        if disk.percent < 90:
            print_success(f"Disk usage: {disk.percent}% (Good)")
        else:
            print_warning(f"Disk usage: {disk.percent}% (High)")
        
        return True
        
    except ImportError:
        print_warning("psutil not available for system monitoring")
        print_info("Install with: pip install psutil")
        return True  # Not critical
    except Exception as e:
        print_warning(f"System health check failed: {e}")
        return True  # Not critical

def generate_validation_report(results):
    """Generate validation report"""
    print_header("VALIDATION SUMMARY")
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    failed_checks = total_checks - passed_checks
    
    success_rate = (passed_checks / total_checks) * 100
    
    print(f"\n{Colors.BOLD}Validation Results:{Colors.END}")
    print(f"  Total Checks: {total_checks}")
    print(f"  Passed: {Colors.GREEN}{passed_checks}{Colors.END}")
    print(f"  Failed: {Colors.RED}{failed_checks}{Colors.END}")
    print(f"  Success Rate: {Colors.BOLD}{success_rate:.1f}%{Colors.END}")
    
    if failed_checks == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL VALIDATIONS PASSED!{Colors.END}")
        print(f"{Colors.GREEN}Your enhanced trading bot is ready to launch!{Colors.END}")
        print(f"\n{Colors.CYAN}Next Steps:{Colors.END}")
        print(f"1. Run: {Colors.BOLD}python src/enhanced_main_execution.py --mode paper{Colors.END}")
        print(f"2. Monitor: {Colors.BOLD}tail -f logs/enhanced_trading.log{Colors.END}")
        print(f"3. Check dashboard at: {Colors.BOLD}http://localhost:8765{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ VALIDATION FAILED{Colors.END}")
        print(f"{Colors.RED}Please fix the issues above before launching the trading bot.{Colors.END}")
        
        failed_items = [name for name, result in results.items() if not result]
        print(f"\n{Colors.YELLOW}Failed Checks:{Colors.END}")
        for item in failed_items:
            print(f"  - {item}")
    
    # Save report to file
    try:
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'success_rate': success_rate,
            'results': results,
            'failed_items': [name for name, result in results.items() if not result]
        }
        
        with open('validation_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n{Colors.CYAN}📄 Detailed report saved to: validation_report.json{Colors.END}")
        
    except Exception as e:
        print_warning(f"Could not save validation report: {e}")

def main():
    """Main validation function"""
    print_header("ENHANCED TRADING BOT - CONFIGURATION VALIDATION")
    print(f"{Colors.CYAN}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass  # Will be checked later
    
    # Run all validation checks
    validation_results = {}
    
    validation_results['python_version'] = check_python_version()
    validation_results['directory_structure'] = check_directory_structure()
    validation_results['core_dependencies'] = check_core_dependencies()
    validation_results['trading_dependencies'] = check_trading_dependencies()
    validation_results['environment_file'] = check_environment_file()
    validation_results['api_keys'] = check_api_keys()
    validation_results['api_connection'] = test_api_connection()
    validation_results['enhanced_modules'] = check_enhanced_modules()
    validation_results['file_permissions'] = check_file_permissions()
    validation_results['database_connection'] = check_database_connection()
    validation_results['system_health'] = run_system_health_check()
    
    # Generate summary report
    generate_validation_report(validation_results)
    
    # Return exit code
    if all(validation_results.values()):
        return 0  # Success
    else:
        return 1  # Failure

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Validation script error: {e}{Colors.END}")
        print(f"{Colors.RED}Traceback:{Colors.END}")
        traceback.print_exc()
        sys.exit(1)