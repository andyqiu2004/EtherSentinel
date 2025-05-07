import asyncio
import logging
from web3 import Web3
from dotenv import load_dotenv
import os
from models.gnn_bert import GNNBERT, TransactionClassifier, AccountAnalyzer
from analysis.transaction_analyzer import TransactionAnalyzer
from analysis.account_analyzer import AccountAnalyzer
from monitoring.realtime_monitor import RealtimeMonitor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EtherSentinel:
    def __init__(self):
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('ETHEREUM_NODE_URL')))
        
        # Initialize models
        self.gnn_bert = GNNBERT(
            num_features=4,  # Adjust based on your feature set
            num_classes=2    # Adjust based on your classification task
        )
        
        # Load trained models
        self._load_models()
        
        # Initialize analyzers
        self.transaction_analyzer = TransactionAnalyzer(self.gnn_bert)
        self.account_analyzer = AccountAnalyzer(self.gnn_bert)
        
        # Initialize real-time monitor
        self.monitor = RealtimeMonitor(
            w3=self.w3,
            transaction_analyzer=self.transaction_analyzer,
            account_analyzer=self.account_analyzer,
            alert_callback=self._handle_alert
        )
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Load transaction classifier
            self.gnn_bert.load_state_dict(
                torch.load('models/checkpoints/gnn_bert_model.pt')
            )
            logger.info("Successfully loaded GNN-BERT model")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    async def _handle_alert(self, alert_data: dict):
        """Handle alerts from the monitoring system"""
        alert_type = alert_data['type']
        
        if alert_type == 'high_risk_transaction':
            logger.warning(f"High risk transaction detected: {alert_data['transaction']['hash']}")
            # TODO: Implement your alert handling logic
            # For example, send notifications, update database, etc.
            
        elif alert_type == 'monitored_transaction':
            logger.info(f"Monitored transaction detected: {alert_data['transaction']['hash']}")
            # TODO: Implement your monitoring logic
    
    async def start(self):
        """Start the EtherSentinel system"""
        logger.info("Starting EtherSentinel...")
        
        # Add addresses to monitor (example)
        self.monitor.add_monitored_address("0x123...")  # Replace with actual addresses
        
        # Start real-time monitoring
        await self.monitor.start_monitoring()
    
    async def stop(self):
        """Stop the EtherSentinel system"""
        logger.info("Stopping EtherSentinel...")
        await self.monitor.stop_monitoring()
    
    def analyze_transaction(self, transaction_hash: str) -> dict:
        """Analyze a specific transaction"""
        try:
            # Get transaction data
            tx = self.w3.eth.get_transaction(transaction_hash)
            
            # Prepare transaction data
            tx_data = {
                'hash': tx['hash'].hex(),
                'from': tx['from'],
                'to': tx['to'],
                'value': self.w3.from_wei(tx['value'], 'ether'),
                'timestamp': self.w3.eth.get_block(tx['blockNumber'])['timestamp']
            }
            
            # Analyze transaction
            return self.transaction_analyzer.analyze_transaction({'transactions': [tx_data]})
            
        except Exception as e:
            logger.error(f"Error analyzing transaction: {str(e)}")
            raise
    
    def analyze_account(self, address: str) -> dict:
        """Analyze a specific account"""
        try:
            # Get account data
            account_data = {
                'address': address,
                'transactions': self._get_account_transactions(address),
                'first_seen': self._get_account_first_seen(address)
            }
            
            # Analyze account
            return self.account_analyzer.analyze_account(account_data)
            
        except Exception as e:
            logger.error(f"Error analyzing account: {str(e)}")
            raise
    
    def _get_account_transactions(self, address: str) -> list:
        """Get transactions for an account"""
        # TODO: Implement transaction fetching logic
        # This could involve querying a blockchain explorer API or your own database
        return []
    
    def _get_account_first_seen(self, address: str) -> int:
        """Get the timestamp of the first transaction for an account"""
        # TODO: Implement first seen timestamp logic
        return 0

async def main():
    # Initialize EtherSentinel
    sentinel = EtherSentinel()
    
    try:
        # Start the system
        await sentinel.start()
        
        # Keep the program running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await sentinel.stop()
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        await sentinel.stop()

if __name__ == "__main__":
    asyncio.run(main()) 