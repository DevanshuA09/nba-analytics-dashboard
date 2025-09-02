import React, { useState, useEffect } from 'react'
import { 
  BarChart3, 
  Target, 
  TrendingUp, 
  Users, 
  MessageSquare, 
  Search,
  Menu,
  X,
  Sun,
  Moon,
  Home,
  User,
  Users2,
  MessageCircle
} from 'lucide-react'
import WinProbCard from './components/WinProbCard'
import PlayerForecastCard from './components/PlayerForecastCard'
import ChatPanel from './components/ChatPanel'
import MatchupExplorer from './components/MatchupExplorer'
import PlayerComparison from './components/PlayerComparison'
import './index.css'

function App() {
  const [games, setGames] = useState([])
  const [teams, setTeams] = useState([])
  const [players, setPlayers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [darkMode, setDarkMode] = useState(false)

  useEffect(() => {
    fetchData()
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('darkMode')
    if (savedTheme) {
      setDarkMode(JSON.parse(savedTheme))
    }
  }, [])

  useEffect(() => {
    // Apply dark mode class to document
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode))
  }, [darkMode])

  const fetchData = async () => {
    try {
      setLoading(true)
      
      // Fetch today's games, teams, and players in parallel
      const [gamesResponse, teamsResponse, playersResponse] = await Promise.all([
        fetch('http://localhost:8000/games/today'),
        fetch('http://localhost:8000/teams'),
        fetch('http://localhost:8000/players')
      ])

      if (!gamesResponse.ok || !teamsResponse.ok || !playersResponse.ok) {
        throw new Error('Failed to fetch data')
      }

      const [gamesData, teamsData, playersData] = await Promise.all([
        gamesResponse.json(),
        teamsResponse.json(),
        playersResponse.json()
      ])

      setGames(gamesData.games || [])
      setTeams(teamsData.teams || [])
      setPlayers(playersData.players || [])
      setError(null)
    } catch (err) {
      console.error('Error fetching data:', err)
      setError('Failed to load data. Please check if the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
  }

  const navigation = [
    { id: 'dashboard', name: 'Dashboard', icon: Home },
    { id: 'chat', name: 'AI Assistant', icon: MessageCircle },
    { id: 'matchup', name: 'Team Matchup', icon: Users2 },
    { id: 'comparison', name: 'Player Compare', icon: User },
  ]

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <div className="space-y-8">
            <WinProbCard 
              games={games}
              teams={teams}
              onRefresh={fetchData}
            />
            <PlayerForecastCard 
              players={players}
              teams={teams}
            />
          </div>
        )
      case 'chat':
        return <ChatPanel />
      case 'matchup':
        return <MatchupExplorer />
      case 'comparison':
        return <PlayerComparison />
      default:
        return null
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-2">
              🏀 NBA Analytics Dashboard
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Loading predictive models and data...
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-gray-600 bg-opacity-75 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200 dark:border-gray-700">
          <h1 className="text-xl font-bold text-gray-800 dark:text-white">
            🏀 NBA Analytics
          </h1>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <nav className="mt-6 px-3">
          {navigation.map((item) => {
            const Icon = item.icon
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id)
                  setSidebarOpen(false)
                }}
                className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-lg mb-1 transition-colors ${
                  activeTab === item.id
                    ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                    : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                }`}
              >
                <Icon className="w-5 h-5 mr-3" />
                {item.name}
              </button>
            )
          })}
        </nav>

        <div className="absolute bottom-4 left-4 right-4">
          <button
            onClick={toggleDarkMode}
            className="w-full flex items-center px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            {darkMode ? <Sun className="w-5 h-5 mr-3" /> : <Moon className="w-5 h-5 mr-3" />}
            {darkMode ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between h-16 px-6">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <Menu className="w-6 h-6" />
            </button>

            <div className="flex items-center space-x-4">
              <h2 className="text-lg font-semibold text-gray-800 dark:text-white">
                {navigation.find(nav => nav.id === activeTab)?.name || 'Dashboard'}
              </h2>
            </div>

            <div className="flex items-center space-x-4">
              <button
                onClick={fetchData}
                className="px-3 py-1 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="p-6">
          {error && (
            <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          {renderContent()}
        </main>
      </div>
    </div>
  )
}

export default App
