import { registerAudioInput } from '/javascripts/audio-input'

let initialized = false

const initialize = () => {
    console.log('Initializing Latopia WebUI')

    registerAudioInput()
}

window.addEventListener('load', () => {
    const interval = setInterval(() => {
        if (document.getElementById('latopia-initialized')) {
            if (initialized) return
            clearInterval(interval)
            initialized = true
            initialize()
        }
    }, 100)
})
