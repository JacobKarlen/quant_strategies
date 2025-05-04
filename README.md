## Interactive dashboard for quantitative investment strategies

Built with Plotly and Dash. Takes an exported Excel file from the saved filter "Excel Nyckeltal" in Börsdata Terminal. Displays tables of the current positions for each of the strategies, where stocks can be deselected and selected to handle certain edge-cases like stocks under take-over or when both the A and B share type of the same stock is selected in a strategy. Also displayds an aggregate view with the total positions of each stock across all strategies.

### Strategies

* **Trending Quality:**

  * Excludes financial sector companies
  * Focuses on Swedish companies with market cap ≥ 500M
  * Excludes companies from NGM and Spotlight exchanges
  * Calculates FCFROE (Free Cash Flow Return on Equity)
  * Ranks companies based on combined scores of ROE, ROA, ROIC, and FCFROE
  * Selects top 40 companies with the highest quality rankings
  * Then selects top 10 companies with highest RS rank from the top 40 companies
* **Trending Growth:**

  * Excludes financial and real estate sectors
  * Focuses on Swedish companies with market cap ≥ 500M
  * Requires positive P/E ratio (currently profitable)
  * Ranks companies based on earnings and revenue growth metrics:
    * 1-year earnings growth
    * 3-year earnings growth
    * 1-year revenue growth
    * 3-year revenue growth
  * Selects top 40 companies with the best combined growth rankings
  * Then selects top 10 companies with highest RS rank from the top 40 companies
* **Trending Value:**

  * Excludes financial sector companies
  * Focuses on Swedish companies with market cap ≥ 500M
  * Excludes companies from NGM and Spotlight exchanges
  * Ranks companies based on multiple valuation metrics:
    * P/E (Price-to-Earnings)
    * P/B (Price-to-Book)
    * P/S (Price-to-Sales)
    * P/FCF (Price-to-Free-Cash-Flow)
    * EV/EBITDA (Enterprise Value to EBITDA)
    * Dividend yield (inverse)
  * Selects top 40 companies with the lowest (best) value rankings
  * Then selects top 10 companies with highest RS rank from the top 40 companies
* **Momentum:**

  * Focuses on Swedish companies with market cap ≥ 500M
  * Excludes companies from NGM and Spotlight exchanges
  * Requires F-Score > 2 (financial strength indicator)
  * Ranks companies based on RS Rank (Relative Strength)
  * Selects top 10 companies with the highest RS Rank
