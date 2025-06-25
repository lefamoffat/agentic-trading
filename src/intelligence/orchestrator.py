#!/usr/bin/env python3
"""Intelligent orchestrator for trading experiment optimization.

This module coordinates the LLM advisor, experiment analyzer, and config
generator to provide intelligent experiment orchestration.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Optional ML tracking integration
try:
    from src.tracking import get_ml_tracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

from src.utils.logger import get_logger
from src.intelligence.experiment_analyzer import ExperimentAnalyzer
from src.intelligence.llm_advisor import LLMAdvisor
from src.intelligence.config_generator import ConfigGenerator
from src.intelligence.reasoning_tracker import ReasoningTracker

class IntelligentOrchestrator:
    """Orchestrates intelligent trading experiment optimization."""
    
    def __init__(self, experiment_name: str = "AgenticTrading"):
        """Initialize intelligent orchestrator."""
        self.logger = get_logger(self.__class__.__name__)
        self.experiment_name = experiment_name
        
        # Initialize components
        self.experiment_analyzer = ExperimentAnalyzer(experiment_name)
        self.llm_advisor = LLMAdvisor()
        self.config_generator = ConfigGenerator()
        self.reasoning_tracker = ReasoningTracker()
        
        self.logger.info(f"Initialized intelligent orchestrator for experiment: {experiment_name}")
    
    def start_new_experiment(
        self,
        symbol: str = "EUR/USD",
        timeframe: str = "1h",
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Start a new trading experiment with LLM-optimized configuration."""
        self.logger.info(f"Starting new experiment optimization for {symbol} ({timeframe})")
        
        # Track reasoning
        self.reasoning_tracker.start_reasoning_session(
            action="start",
            experiment_id=None,
            llm_model=self.llm_advisor.model
        )
        
        try:
            # Analyze historical experiments
            experiment_data = self.experiment_analyzer.summarize_for_llm(
                action="start",
                symbol=symbol,
                days=30
            )
            
            # Log input data for reasoning
            self.reasoning_tracker.log_input_data(experiment_data)
            
            # Get LLM recommendations
            recommendations = self.llm_advisor.get_recommendations(
                action="start",
                experiment_data=experiment_data,
                symbol=symbol
            )
            
            # Log LLM reasoning
            self.reasoning_tracker.log_llm_reasoning(
                reasoning=recommendations.get("reasoning", ""),
                confidence_score=recommendations.get("confidence_score", 0.0),
                fallback_used=recommendations.get("fallback_used", False)
            )
            self.reasoning_tracker.log_recommended_changes(
                recommendations.get("recommended_changes", {})
            )
            
            # Generate configuration
            trading_config = self.config_generator.generate_config(
                recommendations=recommendations,
                action="start"
            )
            
            result = {
                "action": "start",
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "llm_recommendations": recommendations,
                "trading_config": trading_config,
            }
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(result, output_dir, "start")
            
            self.logger.info("Successfully generated new experiment configuration")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to start new experiment: {e}")
            raise
        
        finally:
            # Finalize reasoning tracking
            self.reasoning_tracker.finalize_reasoning_session()
    
    def continue_experiment(
        self,
        run_id: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Continue an existing experiment with LLM analysis."""
        self.logger.info(f"Analyzing experiment {run_id} for continuation")
        
        # Track reasoning
        self.reasoning_tracker.start_reasoning_session(
            action="continue",
            experiment_id=run_id,
            llm_model=self.llm_advisor.model
        )
        
        try:
            # Analyze current experiment
            experiment_data = self.experiment_analyzer.summarize_for_llm(
                action="continue",
                current_run_id=run_id,
                days=30
            )
            
            # Log input data for reasoning
            self.reasoning_tracker.log_input_data(experiment_data)
            
            # Get LLM recommendations
            recommendations = self.llm_advisor.get_recommendations(
                action="continue",
                experiment_data=experiment_data
            )
            
            # Log LLM reasoning
            self.reasoning_tracker.log_llm_reasoning(
                reasoning=recommendations.get("reasoning", ""),
                confidence_score=recommendations.get("confidence_score", 0.0),
                fallback_used=recommendations.get("fallback_used", False)
            )
            self.reasoning_tracker.log_recommended_changes(
                recommendations.get("recommended_changes", {})
            )
            
            result = {
                "action": "continue",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "action_recommendation": recommendations.get("action_recommendation", "continue"),
                "llm_recommendations": recommendations,
            }
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(result, output_dir, "continue")
            
            self.logger.info(f"Analysis complete. Recommendation: {result['action_recommendation']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze experiment for continuation: {e}")
            raise
        
        finally:
            # Finalize reasoning tracking
            self.reasoning_tracker.finalize_reasoning_session()
    
    def _save_results(
        self,
        result: Dict[str, Any],
        output_dir: Path,
        action: str
    ) -> None:
        """Save orchestration results to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intelligence_{action}_{timestamp}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")

def main():
    """Main CLI entry point for intelligent orchestrator."""
    parser = argparse.ArgumentParser(description="Intelligent Trading Experiment Orchestrator")
    parser.add_argument("action", choices=["start", "continue", "optimize", "analyze"], 
                       help="Action to perform")
    parser.add_argument("--symbol", default="EUR/USD", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Data timeframe") 
    parser.add_argument("--run-id", help="Experiment run ID (for continue/optimize)")
    parser.add_argument("--metric", default="sharpe_ratio", help="Metric to optimize")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--experiment-name", default="AgenticTrading", help="Experiment name")
    parser.add_argument("--llm-model", default="gpt-4", help="LLM model to use")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning tracking")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = IntelligentOrchestrator(
        experiment_name=args.experiment_name,
        llm_model=args.llm_model,
        enable_reasoning_tracking=not args.no_reasoning
    )
    
    try:
        if args.action == "start":
            result = orchestrator.start_new_experiment(
                symbol=args.symbol,
                timeframe=args.timeframe,
                output_dir=args.output_dir
            )
            print(f"✅ New experiment configuration generated")
            print(f"Confidence: {result['llm_recommendations'].get('confidence_score', 0):.2f}")
            
        elif args.action == "continue":
            if not args.run_id:
                print("❌ --run-id required for continue action")
                sys.exit(1)
            
            result = orchestrator.continue_experiment(
                run_id=args.run_id,
                output_dir=args.output_dir
            )
            recommendation = result.get('action_recommendation', 'unknown')
            print(f"✅ Experiment analysis complete")
            print(f"Recommendation: {recommendation}")
            
        elif args.action == "optimize":
            result = orchestrator.optimize_experiment(
                run_id=args.run_id,
                metric=args.metric,
                output_dir=args.output_dir
            )
            print(f"✅ Optimization analysis complete")
            print(f"Strategy: {result.get('optimization_strategy', 'general')}")
            
        elif args.action == "analyze":
            summary = orchestrator.get_reasoning_summary(days=30)
            print(f"✅ Reasoning analysis complete")
            print(f"Recent decisions: {len(summary.get('recent_reasoning', []))}")
        
        # Print reasoning if available
        if "llm_recommendations" in locals() and "result" in locals():
            reasoning = result.get("llm_recommendations", {}).get("reasoning", "")
            if reasoning:
                print(f"\n💭 LLM Reasoning:\n{reasoning}")
    
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 