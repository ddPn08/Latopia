export const registerAudioInput = () => {
    for (const element of Array.from(document.getElementsByClassName('latopia-audio-input'))) {
        element.addEventListener('change', async (e) => {
            /**
             * @type {HTMLInputElement}
             */
            const input = e.target
            if (input.files.length === 0) return
            const file = input.files[0]

            const buttons = document.getElementsByClassName('latopia-infer-button')

            input.disabled = true
            for (const button of buttons) button.disabled = true

            const form = new FormData()
            form.append('file', file)
            try {
                await fetch('/api/infer/input-file-upload', {
                    method: 'POST',
                    body: form,
                })
            } catch (error) {
                console.error(error)
            }

            input.disabled = false
            for (const button of buttons) button.disabled = false
        })
    }
}
