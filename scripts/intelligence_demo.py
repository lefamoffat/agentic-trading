#!/usr/bin/env python3
"""Demo script for the intelligent trading system."""
import argparse
import json
from pathlib import Path

from src.intelligence import IntelligentOrchestrator


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Intelligent Trading System Demo")
    parser.add_argument("action", choices=["start", "continue"], help="Action to perform")
    parser.add_argument("--symbol", default="EUR/USD", help="Trading symbol")
    parser.add_argument("--run-id", help="MLflow run ID (for continue)")
    parser.add_argument("--output", type=Path, help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🧠 Intelligent Trading System Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = IntelligentOrchestrator()
    
    try:
        if args.action == "start":
            print(f"🚀 Starting new experiment for {args.symbol}")
            result = orchestrator.start_new_experiment(
                symbol=args.symbol,
                output_dir=args.output
            )
            
            print(f"✅ Configuration generated!")
            print(f"📊 Confidence: {result['llm_recommendations']['confidence_score']:.2f}")
            print(f"💰 Initial Balance: ${result['trading_config'].initial_balance:,.2f}")
            
            # Show LLM reasoning
            reasoning = result['llm_recommendations']['reasoning']
            print(f"\n💭 LLM Reasoning:")
            print(f"   {reasoning}")
            
            # Show recommended changes
            changes = result['llm_recommendations']['recommended_changes']
            if changes:
                print(f"\n🔧 Recommended Changes:")
                for param, info in changes.items():
                    print(f"   • {param}: {info['value']} ({info['reason']})")
        
        elif args.action == "continue":
            if not args.run_id:
                print("❌ --run-id required for continue action")
                return
            
            print(f"🔄 Analyzing experiment {args.run_id}")
            result = orchestrator.continue_experiment(
                run_id=args.run_id,
                output_dir=args.output
            )
            
            recommendation = result['action_recommendation']
            print(f"✅ Analysis complete!")
            print(f"📋 Recommendation: {recommendation}")
            
            # Show LLM reasoning
            reasoning = result['llm_recommendations']['reasoning']
            print(f"\n💭 LLM Reasoning:")
            print(f"   {reasoning}")
        
        # Show output location if specified
        if args.output:
            print(f"\n📁 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("\n🎯 Demo completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main()) 