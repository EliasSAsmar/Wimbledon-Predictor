// Tennis Players Database (Mock Data)
const TENNIS_PLAYERS = [
    'Novak Djokovic', 'Carlos Alcaraz', 'Daniil Medvedev', 'Jannik Sinner',
    'Andrey Rublev', 'Stefanos Tsitsipas', 'Holger Rune', 'Casper Ruud',
    'Taylor Fritz', 'Alex de Minaur', 'Grigor Dimitrov', 'Tommy Paul',
    'Ben Shelton', 'Cameron Norrie', 'Ugo Humbert', 'Lorenzo Musetti',
    'Sebastian Baez', 'Matteo Berrettini', 'Karen Khachanov', 'Frances Tiafoe',
    'Rafael Nadal', 'Roger Federer', 'Andy Murray', 'Stan Wawrinka',
    'Dominic Thiem', 'Alexander Zverev', 'Marin Cilic', 'Denis Shapovalov',
    'Felix Auger-Aliassime', 'Hubert Hurkacz', 'Diego Schwartzman'
];

// Global state
let selectedPlayers = {
    player1: null,
    player2: null
};

// DOM Elements
const player1Input = document.getElementById('player1-input');
const player2Input = document.getElementById('player2-input');
const player1Dropdown = document.getElementById('player1-dropdown');
const player2Dropdown = document.getElementById('player2-dropdown');
const player1Info = document.getElementById('player1-info');
const player2Info = document.getElementById('player2-info');
const predictBtn = document.getElementById('predict-btn');
const predictionResults = document.getElementById('prediction-results');

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    animateOnScroll();
    initializeLiveBoard();
});

// Event Listeners
function initializeEventListeners() {
    // Player search functionality
    player1Input.addEventListener('input', (e) => handlePlayerSearch(e, 1));
    player2Input.addEventListener('input', (e) => handlePlayerSearch(e, 2));
    
    // Click outside to close dropdowns
    document.addEventListener('click', handleOutsideClick);
    
    // Predict button
    predictBtn.addEventListener('click', generatePrediction);
    
    // Model tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => switchModel(e.target.dataset.model));
    });
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Player search functionality
function handlePlayerSearch(event, playerNumber) {
    const query = event.target.value.toLowerCase();
    const dropdown = playerNumber === 1 ? player1Dropdown : player2Dropdown;
    
    if (query.length < 2) {
        dropdown.style.display = 'none';
        return;
    }
    
    const filteredPlayers = TENNIS_PLAYERS.filter(player => 
        player.toLowerCase().includes(query)
    ).slice(0, 5);
    
    if (filteredPlayers.length > 0) {
        displayPlayerDropdown(filteredPlayers, dropdown, playerNumber);
    } else {
        dropdown.style.display = 'none';
    }
}

// Display player dropdown
function displayPlayerDropdown(players, dropdown, playerNumber) {
    dropdown.innerHTML = '';
    
    players.forEach(player => {
        const playerOption = document.createElement('div');
        playerOption.className = 'player-option';
        playerOption.style.cssText = `
            padding: 12px 15px;
            cursor: pointer;
            border-bottom: 1px solid #e9ecef;
            transition: background-color 0.2s;
        `;
        playerOption.textContent = player;
        
        playerOption.addEventListener('mouseenter', () => {
            playerOption.style.backgroundColor = '#f8f9fa';
        });
        
        playerOption.addEventListener('mouseleave', () => {
            playerOption.style.backgroundColor = 'transparent';
        });
        
        playerOption.addEventListener('click', () => {
            selectPlayer(player, playerNumber);
        });
        
        dropdown.appendChild(playerOption);
    });
    
    dropdown.style.display = 'block';
}

// Select player
function selectPlayer(playerName, playerNumber) {
    const input = playerNumber === 1 ? player1Input : player2Input;
    const dropdown = playerNumber === 1 ? player1Dropdown : player2Dropdown;
    const info = playerNumber === 1 ? player1Info : player2Info;
    
    input.value = playerName;
    dropdown.style.display = 'none';
    
    selectedPlayers[`player${playerNumber}`] = playerName;
    
    updatePlayerInfo(playerName, info);
    updatePredictButton();
}

// Update player info display
function updatePlayerInfo(playerName, infoElement) {
    const avatar = infoElement.querySelector('.player-avatar');
    const details = infoElement.querySelector('.player-details');
    
    // Add player initials to avatar
    const initials = playerName.split(' ').map(name => name[0]).join('');
    avatar.innerHTML = initials;
    
    // Update player details
    details.innerHTML = `
        <h3>${playerName}</h3>
        <p>Professional Tennis Player</p>
    `;
    
    // Add animation
    infoElement.style.animation = 'fadeInUp 0.5s ease-out';
}

// Update predict button state
function updatePredictButton() {
    if (selectedPlayers.player1 && selectedPlayers.player2) {
        predictBtn.disabled = false;
        predictBtn.style.opacity = '1';
    } else {
        predictBtn.disabled = true;
        predictBtn.style.opacity = '0.5';
    }
}

// Handle clicks outside dropdowns
function handleOutsideClick(event) {
    if (!event.target.closest('.player-search')) {
        player1Dropdown.style.display = 'none';
        player2Dropdown.style.display = 'none';
    }
}

// Generate prediction (mock functionality)
function generatePrediction() {
    if (!selectedPlayers.player1 || !selectedPlayers.player2) return;
    
    // Show loading state
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    predictBtn.disabled = true;
    
    // Simulate API call delay
    setTimeout(() => {
        const prediction = generateMockPrediction();
        displayPredictionResults(prediction);
        
        // Reset button
        predictBtn.innerHTML = '<i class="fas fa-brain"></i> Generate Prediction';
        predictBtn.disabled = false;
        
        // Show results with animation
        predictionResults.style.display = 'block';
        predictionResults.style.animation = 'fadeInUp 0.6s ease-out';
        
        // Animate probability bar
        setTimeout(() => {
            animateProbabilityBar(prediction.player1WinProb);
        }, 300);
        
    }, 2000);
}

// Generate mock prediction data
function generateMockPrediction() {
    // Generate realistic probabilities
    const player1WinProb = 0.3 + Math.random() * 0.4; // Between 30% and 70%
    const player2WinProb = 1 - player1WinProb;
    
    return {
        player1: selectedPlayers.player1,
        player2: selectedPlayers.player2,
        player1WinProb: player1WinProb,
        player2WinProb: player2WinProb,
        models: {
            xgboost: {
                accuracy: 0.652,
                player1Prob: player1WinProb + (Math.random() - 0.5) * 0.1,
                confidence: 0.85 + Math.random() * 0.1
            },
            randomForest: {
                accuracy: 0.648,
                player1Prob: player1WinProb + (Math.random() - 0.5) * 0.1,
                confidence: 0.82 + Math.random() * 0.1
            }
        }
    };
}

// Display prediction results
function displayPredictionResults(prediction) {
    // Update probability labels
    document.getElementById('player1-prob').textContent = 
        Math.round(prediction.player1WinProb * 100) + '%';
    document.getElementById('player2-prob').textContent = 
        Math.round(prediction.player2WinProb * 100) + '%';
    
    // Update betting odds
    updateBettingOdds(prediction);
    
    // Update feature bars with animation
    animateFeatureBars();
}

// Animate probability bar
function animateProbabilityBar(player1Prob) {
    const probFill = document.getElementById('prob-fill');
    const percentage = player1Prob * 100;
    
    probFill.style.width = percentage + '%';
    
    // Update gradient based on probability
    if (percentage > 60) {
        probFill.style.background = 'linear-gradient(135deg, #00b359, #7fcc7f)';
    } else if (percentage < 40) {
        probFill.style.background = 'linear-gradient(135deg, #ff6b6b, #ff8e8e)';
    } else {
        probFill.style.background = 'linear-gradient(135deg, #ffd700, #ffed4a)';
    }
}

// Update betting odds
function updateBettingOdds(prediction) {
    const player1Prob = prediction.player1WinProb;
    const player2Prob = prediction.player2WinProb;
    
    // Calculate American odds
    const player1American = player1Prob > 0.5 ? 
        `-${Math.round(100 * player1Prob / (1 - player1Prob))}` : 
        `+${Math.round(100 * (1 - player1Prob) / player1Prob)}`;
    
    const player2American = player2Prob > 0.5 ? 
        `-${Math.round(100 * player2Prob / (1 - player2Prob))}` : 
        `+${Math.round(100 * (1 - player2Prob) / player2Prob)}`;
    
    // Calculate decimal odds
    const player1Decimal = (1 / player1Prob).toFixed(2);
    const player2Decimal = (1 / player2Prob).toFixed(2);
    
    // Calculate fractional odds
    const player1Fractional = calculateFractionalOdds(player1Prob);
    const player2Fractional = calculateFractionalOdds(player2Prob);
    
    // Update DOM
    document.getElementById('american-1').textContent = player1American;
    document.getElementById('american-2').textContent = player2American;
    document.getElementById('decimal-1').textContent = player1Decimal;
    document.getElementById('decimal-2').textContent = player2Decimal;
    document.getElementById('fractional-1').textContent = player1Fractional;
    document.getElementById('fractional-2').textContent = player2Fractional;
}

// Calculate fractional odds
function calculateFractionalOdds(probability) {
    const decimal = 1 / probability;
    const numerator = Math.round((decimal - 1) * 10);
    const denominator = 10;
    
    // Simplify fraction
    const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
    const divisor = gcd(numerator, denominator);
    
    return `${numerator / divisor}/${denominator / divisor}`;
}

// Animate feature importance bars
function animateFeatureBars() {
    const featureFills = document.querySelectorAll('.feature-fill');
    
    featureFills.forEach((fill, index) => {
        setTimeout(() => {
            const randomWidth = 50 + Math.random() * 40; // Between 50% and 90%
            fill.style.width = randomWidth + '%';
        }, index * 200);
    });
}

// Switch between models
function switchModel(modelType) {
    const tabBtns = document.querySelectorAll('.tab-btn');
    
    tabBtns.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.model === modelType) {
            btn.classList.add('active');
        }
    });
    
    // Update results based on model (mock functionality)
    // In real implementation, this would fetch different model results
    console.log(`Switched to ${modelType} model`);
}

// Smooth scroll function
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Animate elements on scroll
function animateOnScroll() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeInUp 0.6s ease-out';
                
                // Animate progress bars when they come into view
                if (entry.target.classList.contains('progress-fill')) {
                    const width = entry.target.style.width || '0%';
                    entry.target.style.width = '0%';
                    setTimeout(() => {
                        entry.target.style.width = width;
                    }, 200);
                }
                
                // Animate importance bars
                if (entry.target.classList.contains('importance-fill')) {
                    const width = entry.target.style.width || '0%';
                    entry.target.style.width = '0%';
                    setTimeout(() => {
                        entry.target.style.width = width;
                    }, 200);
                }
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    // Observe elements for animation
    const animatedElements = document.querySelectorAll(
        '.stat-card, .about-feature, .diagram-node, .progress-fill, .importance-fill'
    );
    
    animatedElements.forEach(el => observer.observe(el));
}

// Add hover effects to interactive elements
document.addEventListener('DOMContentLoaded', function() {
    // Add hover effects to player cards
    const playerCards = document.querySelectorAll('.player-card');
    playerCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 8px 25px rgba(0, 102, 51, 0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = 'none';
        });
    });
    
    // Add pulse animation to predict button when enabled
    const observePredictBtn = () => {
        if (!predictBtn.disabled) {
            predictBtn.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-3px) scale(1.02)';
            });
            
            predictBtn.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
            });
        }
    };
    
    // Monitor predict button state
    const observer = new MutationObserver(observePredictBtn);
    observer.observe(predictBtn, { attributes: true, attributeFilter: ['disabled'] });
    observePredictBtn();
});

// Add parallax effect to hero background
window.addEventListener('scroll', function() {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero-background');
    if (hero) {
        hero.style.transform = `translateY(${scrolled * 0.5}px)`;
    }
});

// Tennis ball floating animation
function createFloatingTennisBalls() {
    const hero = document.querySelector('.hero');
    if (!hero) return;
    
    for (let i = 0; i < 3; i++) {
        const ball = document.createElement('div');
        ball.innerHTML = '<i class="fas fa-tennis-ball"></i>';
        ball.style.cssText = `
            position: absolute;
            color: rgba(255, 255, 255, 0.1);
            font-size: ${20 + Math.random() * 30}px;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: float ${5 + Math.random() * 5}s ease-in-out infinite;
            animation-delay: ${Math.random() * 2}s;
            pointer-events: none;
            z-index: 1;
        `;
        hero.appendChild(ball);
    }
}

// Add floating animation CSS
const floatingAnimation = `
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
`;

const style = document.createElement('style');
style.textContent = floatingAnimation;
document.head.appendChild(style);

// Initialize floating tennis balls
document.addEventListener('DOMContentLoaded', createFloatingTennisBalls);

// Add typewriter effect to hero title (optional enhancement)
function typewriterEffect() {
    const titleElement = document.querySelector('.hero-title');
    if (!titleElement) return;
    
    const originalText = titleElement.innerHTML;
    titleElement.innerHTML = '';
    titleElement.style.borderRight = '2px solid white';
    
    let index = 0;
    const speed = 50;
    
    function typeChar() {
        if (index < originalText.length) {
            titleElement.innerHTML += originalText.charAt(index);
            index++;
            setTimeout(typeChar, speed);
        } else {
            titleElement.style.borderRight = 'none';
        }
    }
    
    // Start typing after a delay
    setTimeout(typeChar, 1000);
}

// Uncomment to enable typewriter effect
// document.addEventListener('DOMContentLoaded', typewriterEffect);

// Live Matches Board Functionality
function initializeLiveBoard() {
    // Initialize filter buttons
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', (e) => handleFilterClick(e.target.dataset.filter));
    });
    
    // Start live updates
    startLiveUpdates();
    
    // Initialize board animations
    animateBoardElements();
}

// Handle filter button clicks
function handleFilterClick(filter) {
    // Update active button
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.filter === filter) {
            btn.classList.add('active');
        }
    });
    
    // Filter matches
    const matchRows = document.querySelectorAll('.match-row');
    matchRows.forEach(row => {
        const status = row.dataset.status;
        let shouldShow = false;
        
        switch (filter) {
            case 'all':
                shouldShow = true;
                break;
            case 'live':
                shouldShow = status === 'live';
                break;
            case 'upcoming':
                shouldShow = status === 'upcoming';
                break;
            case 'value':
                // Show rows with positive edge
                const edge = row.querySelector('.edge');
                shouldShow = edge && edge.classList.contains('positive');
                break;
        }
        
        if (shouldShow) {
            row.style.display = 'grid';
            row.style.animation = 'fadeInUp 0.5s ease-out';
        } else {
            row.style.display = 'none';
        }
    });
}

// Start live updates for the board
function startLiveUpdates() {
    // Update live scores every 30 seconds
    setInterval(updateLiveScores, 30000);
    
    // Update odds every 60 seconds
    setInterval(updateOdds, 60000);
    
    // Update confidence bars animation
    setInterval(animateConfidenceBars, 45000);
}

// Mock function to update live scores
function updateLiveScores() {
    const liveMatches = document.querySelectorAll('.match-row[data-status="live"]');
    
    liveMatches.forEach(match => {
        const scoreElement = match.querySelector('.score');
        if (scoreElement && scoreElement.textContent !== 'Not Started') {
            // Simulate score updates
            const scores = [
                '6-4, 3-3', '6-4, 4-2', '6-4, 4-3', '6-4, 5-3',
                '6-4, 6-4', '6-4, 2-6, 1-0', '6-4, 2-6, 2-1'
            ];
            const randomScore = scores[Math.floor(Math.random() * scores.length)];
            scoreElement.textContent = randomScore;
            
            // Add update animation
            scoreElement.style.animation = 'pulse 0.5s ease-out';
            setTimeout(() => {
                scoreElement.style.animation = '';
            }, 500);
        }
    });
}

// Mock function to update odds
function updateOdds() {
    const oddsElements = document.querySelectorAll('.odds-value');
    
    oddsElements.forEach(element => {
        if (Math.random() > 0.7) { // 30% chance to update each odds
            const currentOdds = parseFloat(element.textContent);
            const change = (Math.random() - 0.5) * 0.2; // Small random change
            const newOdds = Math.max(1.01, currentOdds + change).toFixed(2);
            
            // Animate the change
            element.style.transform = 'scale(1.1)';
            element.style.background = '#ffd700';
            
            setTimeout(() => {
                element.textContent = newOdds;
                element.style.transform = 'scale(1)';
                element.style.background = '';
            }, 300);
        }
    });
    
    // Recalculate edges after odds update
    setTimeout(updateEdges, 500);
}

// Update edge calculations
function updateEdges() {
    const matchRows = document.querySelectorAll('.match-row');
    
    matchRows.forEach(row => {
        const marketOdds = row.querySelectorAll('.market-odds .odds-value');
        const asmarOdds = row.querySelectorAll('.asmar-odds .odds-value');
        const edgeElement = row.querySelector('.edge span');
        
        if (marketOdds.length >= 2 && asmarOdds.length >= 2) {
            const marketOdd1 = parseFloat(marketOdds[0].textContent);
            const asmarOdd1 = parseFloat(asmarOdds[0].textContent);
            
            // Calculate edge (simplified)
            const edge = ((marketOdd1 - asmarOdd1) / asmarOdd1) * 100;
            const edgeFormatted = (edge > 0 ? '+' : '') + edge.toFixed(1) + '%';
            
            edgeElement.textContent = edgeFormatted;
            
            // Update edge color
            const edgeContainer = row.querySelector('.edge');
            if (edge > 0) {
                edgeContainer.className = 'edge positive';
            } else {
                edgeContainer.className = 'edge negative';
            }
        }
    });
}

// Animate confidence bars
function animateConfidenceBars() {
    const confidenceFills = document.querySelectorAll('.confidence-fill');
    
    confidenceFills.forEach(fill => {
        const currentWidth = parseInt(fill.style.width);
        const change = Math.floor((Math.random() - 0.5) * 10); // Random change of ±5%
        const newWidth = Math.max(50, Math.min(95, currentWidth + change));
        
        fill.style.width = newWidth + '%';
        fill.parentElement.querySelector('.confidence-text').textContent = newWidth + '%';
    });
}

// Animate board elements on load
function animateBoardElements() {
    const matchRows = document.querySelectorAll('.match-row');
    const summaryCards = document.querySelectorAll('.summary-card');
    
    // Animate match rows
    matchRows.forEach((row, index) => {
        setTimeout(() => {
            row.style.animation = 'fadeInUp 0.6s ease-out';
        }, index * 100);
    });
    
    // Animate summary cards
    summaryCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.animation = 'fadeInUp 0.6s ease-out';
        }, 200 + index * 100);
    });
    
    // Animate confidence bars
    setTimeout(() => {
        const confidenceFills = document.querySelectorAll('.confidence-fill');
        confidenceFills.forEach((fill, index) => {
            setTimeout(() => {
                const width = fill.style.width;
                fill.style.width = '0%';
                setTimeout(() => {
                    fill.style.width = width;
                }, 100);
            }, index * 150);
        });
    }, 500);
}

// Add pulsing effect to live indicators
setInterval(() => {
    const liveIndicators = document.querySelectorAll('.status-badge.live i');
    liveIndicators.forEach(indicator => {
        indicator.style.animation = 'none';
        setTimeout(() => {
            indicator.style.animation = 'pulse 1s infinite';
        }, 10);
    });
}, 2000);

// Real-time board updates simulation
function simulateRealTimeUpdates() {
    setInterval(() => {
        // Random chance to update various elements
        if (Math.random() > 0.8) updateLiveScores();
        if (Math.random() > 0.9) updateOdds();
        if (Math.random() > 0.85) {
            // Update summary statistics
            const summaryNumbers = document.querySelectorAll('.summary-content h3');
            summaryNumbers.forEach(number => {
                if (Math.random() > 0.7) {
                    const current = parseInt(number.textContent);
                    if (!isNaN(current)) {
                        const change = Math.floor((Math.random() - 0.5) * 4);
                        number.textContent = Math.max(0, current + change);
                        number.style.color = '#ffd700';
                        setTimeout(() => {
                            number.style.color = '';
                        }, 1000);
                    }
                }
            });
        }
    }, 10000); // Update every 10 seconds
}

// Start real-time simulation
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(simulateRealTimeUpdates, 5000); // Start after 5 seconds
}); 