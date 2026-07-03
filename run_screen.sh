#!/bin/bash
# Main screening runner that automatically activates virtual environment

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Show usage if no arguments or --help is passed
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  screen     Run the quant screening engine (default)"
    echo "  ticker     Check status of a specific ticker"
    echo "  scan       Run the optimized full-market parallel scan"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run full screen with config.yaml"
    echo "  $0 screen --tickers AAPL MSFT NVDA"
    echo "  $0 ticker AAPL              # Check status of AAPL"
    echo "  $0 ticker NVDA --use-fmp    # Check NVDA with FMP fundamentals"
    echo "  $0 scan --test-mode         # Run optimized scan in test mode"
    echo "  $0 scan --conservative      # Run safe parallel scan"
    exit 0
fi

COMMAND="${1:-screen}"

case "$COMMAND" in
    ticker)
        shift
        python check_ticker.py "$@"
        ;;
    scan)
        shift
        python run_optimized_scan.py "$@"
        ;;
    screen|*)
        # Default: run the quant engine (supports --tickers, --config, etc.)
        if [ "$COMMAND" != "screen" ]; then
            # No recognized command – pass everything as arguments to screen
            python scripts/run_quant_engine.py "$@"
        else
            shift
            python scripts/run_quant_engine.py "$@"
        fi
        ;;
esac
