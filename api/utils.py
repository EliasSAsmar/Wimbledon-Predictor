def american_to_implied_probability(american_odds):
    """
    Convert American odds to implied probability
    
    Args:
        american_odds (int): American odds (e.g., +150, -200)
        
    Returns:
        float: Implied probability (0.0 to 1.0)
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def calculate_ev(our_probability, implied_probability):
    """
    Calculate expected value (EV)
    
    Args:
        our_probability (float): Our model's probability (0.0 to 1.0)
        implied_probability (float): Sportsbook's implied probability (0.0 to 1.0)
        
    Returns:
        float: Expected value as a percentage
    """
    return (our_probability - implied_probability) * 100

def is_qualifying_play(ev_percentage, threshold=7.0):
    """
    Determine if a play meets the EV threshold
    
    Args:
        ev_percentage (float): Expected value as a percentage
        threshold (float): Minimum EV threshold
        
    Returns:
        bool: True if play meets threshold
    """
    return ev_percentage >= threshold

def format_american_odds(odds):
    """
    Format American odds with + or - sign
    
    Args:
        odds (int): Raw odds value
        
    Returns:
        str: Formatted odds (e.g., "+150", "-200")
    """
    return f"{'+' if odds > 0 else ''}{odds}" 