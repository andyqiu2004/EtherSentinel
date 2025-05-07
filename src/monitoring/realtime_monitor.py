import asyncio
from typing import Dict, List, Callable
from web3 import Web3
from datetime import datetime
import json
import logging
from ..analysis.transaction_analyzer import TransactionAnalyzer
from ..analysis.account_analyzer import AccountAnalyzer

class RealtimeMonitor:
    def __init__(self, 
                 w3: Web3,
                 transaction_analyzer: TransactionAnalyzer,
                 account_analyzer: AccountAnalyzer,
                 alert_callback: Callable = None):
        self.w3 = w3
        self.transaction_analyzer = transaction_analyzer
        self.account_analyzer = account_analyzer
        self.alert_callback = alert_callback
        self.monitored_addresses = set()
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    async def start_monitoring(self):
        """Start monitoring blockchain for transactions"""
        self.running = True
        self.logger.info("Starting real-time monitoring...")
        
        # Subscribe to new blocks
        async for block in self.w3.eth.subscribe('newHeads'):
            if not self.running:
                break
                
            await self._process_block(block)
    
    async def stop_monitoring(self):
        """Stop the monitoring process"""
        self.running = False
        self.logger.info("Stopping real-time monitoring...")
    
    async def _process_block(self, block):
        """Process a new block and analyze its transactions"""
        block_number = block['number']
        self.logger.info(f"Processing block {block_number}")
        
        # Get block details
        block_data = self.w3.eth.get_block(block_number, full_transactions=True)
        
        for tx in block_data.transactions:
            await self._analyze_transaction(tx)
    
    async def _analyze_transaction(self, transaction):
        """Analyze a single transaction"""
        try:
            # Prepare transaction data
            tx_data = {
                'hash': transaction['hash'].hex(),
                'from': transaction['from'],
                'to': transaction['to'],
                'value': self.w3.from_wei(transaction['value'], 'ether'),
                'timestamp': datetime.now().timestamp()
            }
            
            # Analyze transaction
            analysis = self.transaction_analyzer.analyze_transaction({'transactions': [tx_data]})
            
            # Check if transaction involves monitored addresses
            if self._is_monitored_transaction(tx_data):
                await self._handle_monitored_transaction(tx_data, analysis)
            
            # Check for high-risk transactions
            if analysis['threat_level'] > 0.8:
                await self._handle_high_risk_transaction(tx_data, analysis)
                
        except Exception as e:
            self.logger.error(f"Error analyzing transaction: {str(e)}")
    
    def _is_monitored_transaction(self, tx_data: Dict) -> bool:
        """Check if transaction involves any monitored addresses"""
        return (tx_data['from'] in self.monitored_addresses or 
                tx_data['to'] in self.monitored_addresses)
    
    async def _handle_monitored_transaction(self, tx_data: Dict, analysis: Dict):
        """Handle transactions involving monitored addresses"""
        if self.alert_callback:
            await self.alert_callback({
                'type': 'monitored_transaction',
                'transaction': tx_data,
                'analysis': analysis
            })
    
    async def _handle_high_risk_transaction(self, tx_data: Dict, analysis: Dict):
        """Handle high-risk transactions"""
        if self.alert_callback:
            await self.alert_callback({
                'type': 'high_risk_transaction',
                'transaction': tx_data,
                'analysis': analysis
            })
    
    def add_monitored_address(self, address: str):
        """Add an address to the monitoring list"""
        self.monitored_addresses.add(address.lower())
        self.logger.info(f"Added address to monitoring: {address}")
    
    def remove_monitored_address(self, address: str):
        """Remove an address from the monitoring list"""
        self.monitored_addresses.discard(address.lower())
        self.logger.info(f"Removed address from monitoring: {address}")
    
    def get_monitored_addresses(self) -> List[str]:
        """Get list of currently monitored addresses"""
        return list(self.monitored_addresses) 